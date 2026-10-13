import ast
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch


class RedisAILibrariesTest(unittest.TestCase):
    def configure(self, child_env):
        tree = ast.parse(Path(__file__).with_name("smartsim_controller.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "configure_redisai_libraries")
        namespace = {"Path": Path, "env": child_env}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "controller", "exec"), namespace)
        return namespace[function.name]

    def test_database_children_use_bundle_without_changing_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "backends/redisai_torch/lib"
            bundle.mkdir(parents=True)
            with patch.dict(os.environ, LD_LIBRARY_PATH="/solver/torch:/cuda:/solver/torch"):
                child_env = os.environ.copy()
                self.configure(child_env)(root / "redisai.so")
                self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/solver/torch:/cuda:/solver/torch")
            self.assertEqual(child_env["LD_LIBRARY_PATH"],
                             f"{bundle}:/solver/torch:/cuda:/solver/torch")
            child_env = {}
            self.configure(child_env)(root / "redisai.so")
            self.assertEqual(child_env["LD_LIBRARY_PATH"], str(bundle))

    def test_missing_bundle_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "bundled Torch libraries not found"):
                self.configure({})(Path(tmp) / "redisai.so")

    def test_failed_start_stops_all_databases_and_drains_only_campaign(self):
        tree = ast.parse(Path(__file__).with_name("smartsim_controller.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "cleanup_failed_start")
        for campaign in (False, True):
            with self.subTest(campaign=campaign):
                child_env = {"SLURM_JOB_ID": "123+0"}
                if campaign:
                    child_env["GPU_REFRESH_CAMPAIGN_ENV"] = "test"
                namespace = {"env": child_env, "sys": sys}
                exec(compile(ast.Module(body=[function], type_ignores=[]), "controller", "exec"), namespace)
                experiment = MagicMock()
                experiment.stop.side_effect = [RuntimeError("unregistered database"), None]
                runner = MagicMock()
                with patch.dict(sys.modules, gpu_refresh_campaign=runner):
                    namespace[function.name](experiment, ["first", "second"])
                self.assertEqual(experiment.stop.call_args_list, [call("first"), call("second")])
                if campaign:
                    runner.clean_steps.assert_called_once_with("123")
                else:
                    runner.clean_steps.assert_not_called()


if __name__ == "__main__":
    unittest.main()
