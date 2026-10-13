
### Application pre/step/post — watercnn (per-call exclusive ms, median of 5 reps)
| region | AIX | PHYDLL_CPP | SS_DEFAULT | SS_TPQ1_I1_B1 | SS_TPQ1_I96_B96 | SS_TPQ12_I8_B96 | SS_TPQ96_I1_B96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `app_prepare_input` | 4.27 | 3.54 | 1.97 | 1.81 | 1.94 | 2.02 | 1.85 |
| `app_provider_inference` | 160.65 | 459.59 | 3984.66 | 9568.56 | 2790.25 | 2147.54 | 855.96 |
| `app_finalize_output` | 0.16 | 0.21 | 0.11 | 0.12 | 0.15 | 0.17 | 0.15 |
| `solver_step_ml_warmup` | 745.36 | 3889.77 | 4541.41 | 10253.82 | 2735.98 | 1517.00 | 927.50 |
| `solver_step_ml_steady` | 165.20 | 463.35 | 3986.86 | 9570.45 | 2792.34 | 2149.57 | 857.97 |

### Provider-internal regions — watercnn (per-call exclusive ms, median of 5 reps)
| region | AIX | PHYDLL_CPP | SS_DEFAULT | SS_TPQ1_I1_B1 | SS_TPQ1_I96_B96 | SS_TPQ12_I8_B96 | SS_TPQ96_I1_B96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `torchInference::forward` | 210.92 | - | - | - | - | - | - |
| `gatherInputData` | 0.00 | - | - | - | - | - | - |
| `scatterOutputData` | 0.00 | - | - | - | - | - | - |
| `aix_inference` | 160.65 | - | - | - | - | - | - |
| `phydll_prepack` | - | 13.80 | - | - | - | - | - |
| `phydll_send` | - | 52.33 | - | - | - | - | - |
| `phydll_recv` | - | 392.31 | - | - | - | - | - |
| `phydll_unpack` | - | 0.11 | - | - | - | - | - |
| `phydll_dl_client` | - | 14616.77 | - | - | - | - | - |
| `dl_torch_forward` | - | 211.63 | - | - | - | - | - |
| `dl_h2d` | - | 4.61 | - | - | - | - | - |
| `dl_d2h` | - | 0.01 | - | - | - | - | - |
| `dl_input_unpack` | - | 6.44 | - | - | - | - | - |
| `dl_output_reorder` | - | 0.33 | - | - | - | - | - |
| `smartsim_put_tensor` | - | - | 263.13 | 292.83 | 178.62 | 155.72 | 147.98 |
| `smartsim_run_model` | - | - | 3438.45 | 8965.44 | 2415.85 | 1803.23 | 551.35 |
| `smartsim_unpack_tensor` | - | - | 0.51 | 7.08 | 16.94 | 9.05 | 3.74 |
| `smartsim_chunk_plan` | - | - | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Application pre/step/post — giant (per-call exclusive ms, median of 5 reps)
| region | AIX | PHYDLL_CPP | SS_DEFAULT | SS_TPQ1_I1_B1 | SS_TPQ1_I96_B96 | SS_TPQ12_I8_B96 | SS_TPQ96_I1_B96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `app_prepare_input` | 0.02 | 0.02 | 0.04 | 0.01 | 0.01 | 0.01 | 0.03 |
| `app_provider_inference` | 3526.53 | 11528.12 | 105669.79 | 200494.79 | 10361.26 | 20154.27 | 8178.28 |
| `app_finalize_output` | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| `solver_step_ml_warmup` | 6395.66 | 14304.90 | 108002.08 | 198359.38 | 10319.50 | 26493.13 | 4410.27 |
| `solver_step_ml_steady` | 3526.57 | 11528.12 | 105669.79 | 200494.79 | 10361.28 | 20154.27 | 8178.33 |

### Provider-internal regions — giant (per-call exclusive ms, median of 5 reps)
| region | AIX | PHYDLL_CPP | SS_DEFAULT | SS_TPQ1_I1_B1 | SS_TPQ1_I96_B96 | SS_TPQ12_I8_B96 | SS_TPQ96_I1_B96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `torchInference::forward` | 3800.29 | - | - | - | - | - | - |
| `gatherInputData` | 0.00 | - | - | - | - | - | - |
| `scatterOutputData` | 0.00 | - | - | - | - | - | - |
| `aix_inference` | 3526.52 | - | - | - | - | - | - |
| `phydll_prepack` | - | 0.05 | - | - | - | - | - |
| `phydll_send` | - | 3.76 | - | - | - | - | - |
| `phydll_recv` | - | 11524.69 | - | - | - | - | - |
| `phydll_unpack` | - | 0.00 | - | - | - | - | - |
| `phydll_dl_client` | - | 136131.25 | - | - | - | - | - |
| `dl_torch_forward` | - | 6646.08 | - | - | - | - | - |
| `dl_h2d` | - | 0.09 | - | - | - | - | - |
| `dl_d2h` | - | 0.03 | - | - | - | - | - |
| `dl_input_unpack` | - | 0.00 | - | - | - | - | - |
| `dl_output_reorder` | - | 0.00 | - | - | - | - | - |
| `smartsim_put_tensor` | - | - | 52.89 | 54.76 | 3.43 | 2.69 | 4.02 |
| `smartsim_run_model` | - | - | 105582.29 | 200386.46 | 10348.27 | 20143.02 | 8170.27 |
| `smartsim_unpack_tensor` | - | - | 0.11 | 0.44 | 5.80 | 2.30 | 0.87 |
| `smartsim_chunk_plan` | - | - | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
