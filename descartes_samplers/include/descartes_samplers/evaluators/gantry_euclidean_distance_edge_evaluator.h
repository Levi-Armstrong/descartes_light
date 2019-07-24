/*
 * Software License Agreement (Apache License)
 *
 * Copyright (c) 2016, Southwest Research Institute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef DESCARTES_SAMPLERS_EVALUATORS_GANTRY_EUCLIDEAN_DISTANCE_EDGE_EVALUATOR_H
#define DESCARTES_SAMPLERS_EVALUATORS_GANTRY_EUCLIDEAN_DISTANCE_EDGE_EVALUATOR_H

#include <descartes_light/interface/edge_evaluator.h>

namespace descartes_light
{

template<typename FloatType>
class GantryEuclideanDistanceEdgeEvaluator : public EdgeEvaluator<FloatType>
{
public:
  GantryEuclideanDistanceEdgeEvaluator();

  bool evaluate(const Rung_<FloatType>& from, const Rung_<FloatType>& to,
                std::vector<typename LadderGraph<FloatType>::EdgeList>& edges) override;
};

using GantryEuclideanDistanceEdgeEvaluatorF = GantryEuclideanDistanceEdgeEvaluator<float>;
using GantryEuclideanDistanceEdgeEvaluatorD = GantryEuclideanDistanceEdgeEvaluator<double>;

} // namespace descartes_light

#endif // DESCARTES_SAMPLERS_EVALUATORS_GANTRY_EUCLIDEAN_DISTANCE_EDGE_EVALUATOR_H
