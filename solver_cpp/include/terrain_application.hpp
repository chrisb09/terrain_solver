#pragma once

#include "application/ml_coupling_application.hpp"

// @registry_name: MLCouplingApplicationTerrainSolver
// @registry_aliases: terrain-solver-app, terrain_solver_app
// @registry_description: Terrain solver coupling application.
template <typename In, typename Out>
class MLCouplingApplicationTerrainSolver : public MLCouplingApplication<In, Out>
{
public:
    MLCouplingApplicationTerrainSolver(MLCouplingData<In> input_data,
                                       MLCouplingData<Out> output_data,
                                       MLCouplingNormalization<In, Out>* normalization)
        : MLCouplingApplication<In, Out>(std::move(input_data), std::move(output_data), normalization) {}

    MLCouplingApplicationTerrainSolver(MLCouplingData<In> input_data,
                                       MLCouplingData<In> input_data_after_preprocessing,
                                       MLCouplingData<Out> output_data_before_postprocessing,
                                       MLCouplingData<Out> output_data,
                                       MLCouplingNormalization<In, Out>* normalization)
        : MLCouplingApplication<In, Out>(std::move(input_data),
                                         std::move(input_data_after_preprocessing),
                                         std::move(output_data_before_postprocessing),
                                         std::move(output_data),
                                         normalization) {}

protected:
    MLCouplingData<In> preprocess(MLCouplingData<In> input_data) override
    {
        return input_data;
    }

    void coupling_step(MLCouplingData<In> input_data_after_preprocessing) override
    {
        (void)input_data_after_preprocessing;
    }

    MLCouplingData<Out> ml_step(MLCouplingData<In> input_data_after_preprocessing) override
    {
        (void)input_data_after_preprocessing;
        return this->output_data_before_postprocessing;
    }

    MLCouplingData<Out> postprocess(MLCouplingData<Out> output_data_before_postprocessing) override
    {
        return output_data_before_postprocessing;
    }
};
