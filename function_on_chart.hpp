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

#ifndef LIFEX_EXAMPLES_FUNCTION_ON_CHART_HPP_
#define LIFEX_EXAMPLES_FUNCTION_ON_CHART_HPP_

#include "core/source/core.hpp"

#include <deal.II/base/parsed_function.h>

#include <memory>

namespace lifex::utils
{
  /**
   * @brief Function defined on a one-dimensional manifold embedded in a
   * @f$d@f$-dimensional space.
   *
   * Let @f$\Omega \subset \mathbb{R}@f$ and @f$ \mathbf{r} : \Omega
   * \rightarrow \mathbb{R}^{d} @f$ be the parametrization of the
   * curve \f$ \Gamma \subset \mathbb{R}^{d} \f$. This class represents a
   * function \f$ F : \Omega \rightarrow \mathbb{R}^{d} \f$.
   *
   * The class is pure virtual. Derived classes must override the
   * @ref value_on_chart method.
   */
  class FunctionOnChart : public Function<dim>
  {
  public:
    /// Dimension of the 1D parametric space.
    inline static constexpr unsigned int chart_dim = 1;
    /// Alias for the geometric parametrization @f$ \mathbf{r} : \Omega \subset
    /// \mathbb{R} \rightarrow \mathbb{R}^{dim} @f$.
    using Embedding = ChartManifold<chart_dim, dim, chart_dim>;

    /// Constructor.
    ///
    /// Links the ChartManifold given in input with the
    /// @ref FunctionOnChart::geometry data member.
    /// Also calls the delegating constructor for a Function<dim>.
    FunctionOnChart(const Embedding &   input_geometry,
                    const unsigned int &input_components = 1);

    /// Destructor.
    virtual ~FunctionOnChart();

    /// @brief Evaluates the function.
    ///
    /// Given an input @f$d@f$-dimensional point @f$ \bar{\mathbf{x}} \in \Gamma
    /// @f$, evaluates its parametric coordinate by inverting @f$\mathbf{r}@f$,
    /// i.e. computes @f$\xi = \mathbf{r}^{-1}(\bar{\mathbf x})@f$, then
    /// evaluates @f$ F\left(\xi\right) @f$.
    ///
    /// @param p_space point in the physical space
    /// @f$ \bar{\mathbf{x}} \in \mathbb{R}^{d}@f$.
    /// @param component @f$ i \leq dim @f$ of the (possibly vector valued)
    /// function @f$ F @f$.
    /// @return value of the function component @f$ F_i @f$ evauated at
    /// @f$\bar{\mathbf{x}}@f$.
    virtual double
    value(const dealii::Point<dim> &p_space,
          unsigned int              component = 0) const override;

  private:
    /// @brief Returns the value of the function evaluated in the 1D parametric
    /// space @f$ F\left(\mathbf{r}(\xi)\right) @f$. If the point does not
    /// belong to the reference domain, an exception is thrown.
    /// @param[in] p_chart point in the 1D reference domain
    /// @f$ \xi \in \Omega @f$.
    /// @param[in] component @f$ i \leq dim @f$ of the (possibly vector valued)
    /// function @f$ F @f$ evaluated on the 1D manifold.
    /// @return value of the function component @f$ F_i @f$ evaluated in the
    /// chart manifold @f$ \mathbf{r}(\xi) @f$.
    virtual double
    value_on_chart(const dealii::Point<chart_dim> &p_chart,
                   const unsigned int &            component) const = 0;

    /// Reference to object representing the embedded geometry @f$ \Gamma
    /// \subset \mathbb{R}^{dim}@f$.
    const Embedding &geometry;
  };


  /**
   * @brief Class that manages the setting and the projection of
   * the parsed function @f$ F @f$ in the reference domain @f$ \Omega @f$.
   */
  class ParsedFunctionOnChart : public FunctionOnChart
  {
  public:
    /// Alias for the parsed function @f$ F @f$
    /// evaluated in the parametric space @f$ \Omega @f$.
    using PtrParsedOnChart =
      std::shared_ptr<Functions::ParsedFunction<chart_dim>>;

    /// Constructor.
    ///
    /// Calls the delegating constructor of FunctionOnChart.
    ParsedFunctionOnChart(const Embedding &   input_domain,
                          const unsigned int &input_components = 1);

    /// @brief Binds the dealii::Functions::ParsedFunction<chart_dim> object given in input,
    /// namely a pointer to the parsed function in the reference domain.
    /// @param[in] ptr_function_parsed pointer to the above mentioned object.
    void
    set_function_on_chart(const PtrParsedOnChart &ptr_function_parsed);

  private:
    /// @brief Returns the value of the parsed function evaluated in the 1D parametric
    /// space @f$ F\left(\mathbf{r}(\xi)\right) @f$. Override of the method
    /// value_on_chart() within the FunctionOnChart class. This method does not
    /// check if the point actually belongs to the chart manifold. Indeed, this
    /// method is expected to work together with Meshhandler1D::pull_back(),
    /// which already ensures that the point correctly belongs to the reference
    /// domain.
    /// @param[in] p_chart point in the 1D reference domain
    /// @f$ \xi \in \Omega @f$.
    /// @param[in] component @f$ i \leq dim @f$ of the (possibly vector valued)
    /// function @f$ F @f$ evaluated on the 1D manifold.
    /// @return value of the function component @f$ F_i @f$ evaluated in the
    /// chart manifold @f$ \mathbf{r}(\xi) @f$.
    double
    value_on_chart(const Point<chart_dim> &p_chart,
                   const unsigned int &    component = 0) const;

    /// Pointer to the parsed function object defined in the reference
    /// domain.
    PtrParsedOnChart ptr_parsed_on_chart;
  };
} // namespace lifex::utils

#endif // LIFEX_EXAMPLES_FUNCTION_ON_CHART_HPP_
