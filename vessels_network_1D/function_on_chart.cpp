/********************************************************************************
  Copyright (C) 2021 - 2023 by the lifex authors.

  This file is part of lifex.

  lifex is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  lifex is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with lifex.  If not, see <http://www.gnu.org/licenses/>.
********************************************************************************/

/**
 * @file
 *
 * @author Michelangelo Gabriele Garroni <michelangelogabriele.garroni@polimi.it>
 * @author Davide Grisi <davide.grisi@polimi.it>
 * @author Giacomo Lorenzon <giacomo.lorenzon@polimi.it>
 */

#include "examples/vessels_network_1D/function_on_chart.hpp"

namespace lifex::utils
{
  FunctionOnChart::FunctionOnChart(const Embedding &   input_geometry,
                                   const unsigned int &input_components)
    : dealii::Function<dim>(input_components)
    , geometry(input_geometry)
  {}

  FunctionOnChart::~FunctionOnChart(){};

  double
  FunctionOnChart::value(const dealii::Point<dim> &p_space,
                         unsigned int              component) const
  {
    const dealii::Point<chart_dim> p_chart = geometry.pull_back(p_space);
    return value_on_chart(p_chart, component);
  };

  ParsedFunctionOnChart::ParsedFunctionOnChart(
    const Embedding &   input_domain,
    const unsigned int &input_components)
    : FunctionOnChart(input_domain, input_components)
  {}

  void
  ParsedFunctionOnChart::set_function_on_chart(
    const PtrParsedOnChart &ptr_function_parsed)
  {
    ptr_parsed_on_chart = ptr_function_parsed;
  }

  double
  ParsedFunctionOnChart::value_on_chart(const Point<chart_dim> &p_chart,
                                        const unsigned int &    component) const
  {
    return ptr_parsed_on_chart->value(p_chart, component);
  };
} // namespace lifex::utils
