/********************************************************************************
  Copyright (C) 2021 - 2023 by the lifex authors.

  This file is part of lifex.

  lifex is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version dim> of the License, or
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

#include "examples/vessels_network_1D/mesh_handler_1D.hpp"

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <utility>

namespace lifex::utils
{
  MeshHandler1D::MeshHandler1D(const std::string &subsection,
                               const bool &       parse_mesh_parameters)
    : CoreModel(subsection)
    , prm_parse_mesh_parameters(parse_mesh_parameters)
  {}

  void
  MeshHandler1D::declare_parameters(ParamHandler &params) const
  {
    // Mesh parameters subsection.
    params.enter_subsection("Mesh and space discretization");
    {
      params.declare_entry_selection("Geometry type", "Linear", "Linear|File");

      params.declare_entry("Number of refinements",
                           "0",
                           Patterns::Integer(0),
                           "Number of uniform refinements of the mesh.");

      if (prm_parse_mesh_parameters)
        {
          params.enter_subsection("Extreme point 0 coordinates");
          {
            params.declare_entry("x", "0.0", Patterns::Double(), "");
            params.declare_entry("y", "0.0", Patterns::Double(), "");
            params.declare_entry("z", "0.0", Patterns::Double(), "");
          }
          params.enter_subsection("Extreme point 1 coordinates");
          {
            params.declare_entry("x", "0.0", Patterns::Double(), "");
            params.declare_entry("y", "0.0", Patterns::Double(), "");
            params.declare_entry("z", "0.0", Patterns::Double(), "");
          }
          params.declare_entry(
            "Non linear geometry file name",
            "",
            Patterns::Anything(),
            "Name of the file where the (eventually) non linear"
            "geometry is saved.");
        }
    }
    params.leave_subsection();
  }

  void
  MeshHandler1D::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read mesh parameters.
    params.enter_subsection("Mesh and space discretization");
    {
      if (params.get("Geometry type") == "Linear")
        prm_geometry_type = GeometryType::Linear;
      else
        prm_geometry_type = GeometryType::File;

      prm_n_refinements = params.get_integer("Number of refinements");

      if (prm_parse_mesh_parameters)
        {
          if (prm_geometry_type == GeometryType::Linear)
            {
              params.enter_subsection("Extreme point 0 coordinates");
              {
                extremal_point_0(0) = params.get_double("x");
                extremal_point_0(1) = params.get_double("y");
                extremal_point_0(2) = params.get_double("z");
              }
              params.enter_subsection("Extreme point 1 coordinates");
              {
                extremal_point_1(0) = params.get_double("x");
                extremal_point_1(1) = params.get_double("y");
                extremal_point_1(2) = params.get_double("z");
              }
            }
          else
            {
              prm_mesh_file_name = params.get("Non linear geometry file name");
            }
        }
    }
    params.leave_subsection();
  }

  void
  MeshHandler1D::initialize_linear_segment(
    const Point<dim> &extremal_space_point_0,
    const Point<dim> &extremal_space_point_1)
  {
    extremal_point_0 = extremal_space_point_0;
    extremal_point_1 = extremal_space_point_1;
    length           = 0.;

    for (unsigned int i = 0; i < dim; ++i)
      {
        direction[i] = extremal_point_1[i] - extremal_point_0[i];
        length += direction[i] * direction[i];
      }

    length = std::sqrt(length);
    direction /= length;
  }

  void
  MeshHandler1D::initialize_linear_segment(
    const Point<dim> &                             extremal_space_point_0,
    double                                         length_,
    const Tensor<FunctionOnChart::chart_dim, dim> &direction_)
  {
    Assert(length_ > 0.0, ExcZero());

    extremal_point_0 = extremal_space_point_0;
    direction        = direction_;
    length           = length_;

    direction /= direction.norm();
    for (unsigned int i = 0; i < dim; ++i)
      {
        extremal_point_1[i] = direction[i] * length + extremal_point_0[i];
      }
  }

  Point<FunctionOnChart::chart_dim>
  MeshHandler1D::pull_back(const Point<dim> &space_point) const
  {
    Assert(prm_geometry_type == GeometryType::Linear,
           ExcNotImplemented()); //@todo: generalize to the non linear case.

    Point<FunctionOnChart::chart_dim> chart_point =
      linear_pull_back(space_point);

    // Project the point onto the reference domain.
    Point<FunctionOnChart::chart_dim> chart_point_constrained = chart_point;
    chart_point_constrained[0] = std::max(-1.0, std::min(1.0, chart_point[0]));

    return chart_point_constrained;
  }

  Point<dim>
  MeshHandler1D::push_forward(
    const Point<FunctionOnChart::chart_dim> &chart_point) const
  {
    Assert(prm_geometry_type == GeometryType::Linear,
           ExcNotImplemented()); //@todo: generalize to the non linear case.

    // Project the point onto the reference domain.
    Point<FunctionOnChart::chart_dim> chart_point_constrained = chart_point;
    chart_point_constrained[0] = std::max(-1.0, std::min(1.0, chart_point[0]));

    return linear_push_forward(chart_point_constrained);
  }

  std::unique_ptr<Manifold<FunctionOnChart::chart_dim, dim>>
  MeshHandler1D::clone() const
  {
    Assert(false, ExcMessage("This class cannot be cloned."));
    return std::make_unique<MeshHandler1D>("");
  }

  double
  MeshHandler1D::project_gradient(
    const Tensor<FunctionOnChart::chart_dim, dim> &gradient,
    const Point<dim> & /*space_point*/) const
  {
    Assert(prm_geometry_type == GeometryType::Linear,
           ExcNotImplemented()); //@todo: generalize to the non linear case.

    return linear_project_gradient(gradient);
  }

  void
  MeshHandler1D::print_info() const
  {
    pcout << "=============================================" << std::endl
          << " Segment's info" << std::endl
          << "=============================================" << std::endl
          << " - Segment length:     " << length << std::endl
          << " - Direction:        (" << direction[0] << ", " << direction[1]
          << ", " << direction[2] << ")" << std::endl
          << " - First extremal point:     (" << extremal_point_0[0] << ", "
          << extremal_point_0[1] << ", " << extremal_point_0[2] << ")"
          << std::endl
          << " - Second extremal point:    (" << extremal_point_1[0] << ", "
          << extremal_point_1[1] << ", " << extremal_point_1[2] << ")"
          << std::endl
          << " - Refinements:       " << prm_n_refinements << std::endl
          << "=============================================" << std::endl
          << std::endl;
  }

  void
  MeshHandler1D::create_mesh()
  {
    if (prm_geometry_type == GeometryType::Linear)
      create_linear_mesh();
    else if (prm_geometry_type == GeometryType::File)
      {
        std::ifstream in(prm_mesh_file_name);

        if (in)
          {
            in.open(prm_mesh_file_name);
            GridIn<FunctionOnChart::chart_dim, dim> grid_in;
            grid_in.attach_triangulation(triangulation);
            grid_in.read_vtk(in);

            triangulation.refine_global(prm_n_refinements);

            length = 0;

            for (const auto &cell : triangulation.cell_iterators())
              length += cell->diameter();
          }
      }
  }

  void
  MeshHandler1D::triangulation_to_vtk() const
  {
    std::ofstream   logfile("triangulation.vtk");
    dealii::GridOut grid_out;
    grid_out.write_vtk(triangulation, logfile);
  }

  Triangulation<FunctionOnChart::chart_dim, dim> &
  MeshHandler1D::get_triangulation()
  {
    return triangulation;
  }

  const Triangulation<FunctionOnChart::chart_dim, dim> &
  MeshHandler1D::get_triangulation() const
  {
    return triangulation;
  }

  std::unique_ptr<QGauss<FunctionOnChart::chart_dim>>
  MeshHandler1D::get_quadrature_gauss(const unsigned int &n_points) const
  {
    return std::make_unique<QGauss<FunctionOnChart::chart_dim>>(n_points);
  }

  std::unique_ptr<FE_Q<FunctionOnChart::chart_dim, dim>>
  MeshHandler1D::get_fe_lagrange(const unsigned int &degree) const
  {
    return std::make_unique<FE_Q<FunctionOnChart::chart_dim, dim>>(degree);
  }

  double
  MeshHandler1D::get_length() const
  {
    Assert(prm_geometry_type == GeometryType::Linear,
           ExcNotImplemented()); //@todo: generalize to the non linear case.

    return length;
  }

  double
  MeshHandler1D::get_n_refinements() const
  {
    return prm_n_refinements;
  }

  Tensor<FunctionOnChart::chart_dim, dim>
  MeshHandler1D::get_direction() const
  {
    Assert(prm_geometry_type == GeometryType::Linear,
           ExcNotImplemented()); //@todo: generalize to the non linear case.

    return direction;
  }

  const Point<dim> &
  MeshHandler1D::get_extremal_point(const unsigned int &point_index) const
  {
    Assert(point_index == 0 || point_index == 1, ExcInvalidState());

    if (point_index == 0)
      return extremal_point_0;
    else // if (point_index == 1)
      return extremal_point_1;
  }

  void
  MeshHandler1D::flag_intervals_for_refinement(const Vector<float> &criterion,
                                               const double &      top_fraction,
                                               const unsigned int &max_n_cells)
  {
    Assert((top_fraction >= 0) && (top_fraction <= 1),
           ExcMessage("Condition not fullfilled: 0 <= top_fraction <= 1."));

    Assert(criterion.is_non_negative(),
           ExcMessage("Negative values are found."));

    Assert(criterion.size() == triangulation.n_active_cells(),
           ExcDimensionMismatch(criterion.size(),
                                triangulation.n_active_cells()));

    // If the criterion values are null, we do not need to refine.
    if (criterion.all_zero())
      return;

    const std::pair<double, double> adjusted_fractions =
      GridRefinement::adjust_refine_and_coarsen_number_fraction<
        FunctionOnChart::chart_dim>(criterion.size(),
                                    max_n_cells,
                                    top_fraction,
                                    0.0);

    const int refine_cells =
      static_cast<int>(adjusted_fractions.first * criterion.size());

    if (refine_cells)
      {
        Vector<double> tmp(criterion);

        if (static_cast<size_t>(refine_cells) == criterion.size())
          this->set_intervals_refinement_flag(
            criterion, std::numeric_limits<double>::lowest(), max_n_cells);
        else
          {
            std::nth_element(tmp.begin(),
                             tmp.begin() + refine_cells - 1,
                             tmp.end(),
                             std::greater<double>());

            this->set_intervals_refinement_flag(
              criterion, *(tmp.begin() + refine_cells - 1), max_n_cells);
          }
      }
  }

  void
  MeshHandler1D::set_intervals_refinement_flag(const Vector<float> &criterion,
                                               double               threshold,
                                               const unsigned int & max_to_mark)
  {
    // If the threshold is zero, find the minimum value among criterion
    // vector.
    if (threshold == 0)
      {
        threshold = criterion(0);
        for (const auto &c_i : criterion)
          if (c_i > 0 && (c_i < threshold))
            threshold = c_i;
      }

    // Count the marked cells.
    unsigned int marked = 0;
    // Actually mark cells.
    for (const auto &cell : triangulation.active_cell_iterators())
      if (criterion(cell->active_cell_index()) >= threshold)
        {
          if (marked >= max_to_mark)
            break;
          ++marked;

          cell->set_refine_flag();
        }
  }

  Point<FunctionOnChart::chart_dim>
  MeshHandler1D::linear_pull_back(const Point<dim> &space_point) const
  {
    Point<FunctionOnChart::chart_dim> p_chart;
    p_chart[0] =
      (direction[0] * space_point[0] + direction[1] * space_point[1] +
       direction[2] * space_point[2] -
       (direction[0] * (extremal_point_0[0] + extremal_point_1[0]) / 2 +
        direction[1] * (extremal_point_0[1] + extremal_point_1[1]) / 2 +
        direction[2] * (extremal_point_0[2] + extremal_point_1[2]) / 2)) /
      length * 2;

    return p_chart;
  }

  Point<dim>
  MeshHandler1D::linear_push_forward(
    const Point<FunctionOnChart::chart_dim> &chart_point) const
  {
    Point<dim>    result;
    const double &xi = chart_point[0];

    for (unsigned int i = 0; i < dim; ++i)
      result[i] = (extremal_point_0[i] + extremal_point_1[i]) / 2 +
                  xi * (extremal_point_1[i] - extremal_point_0[i]) / 2;

    return result;
  }

  double
  MeshHandler1D::linear_project_gradient(
    const Tensor<FunctionOnChart::chart_dim, dim> &gradient) const
  {
    double directional_derivative{0.0};
    for (unsigned i = 0; i < dim; ++i)
      directional_derivative += gradient[i] * direction[i];
    return directional_derivative;
  }

  void
  MeshHandler1D::create_linear_mesh()
  {
    initialize_linear_segment(extremal_point_0, extremal_point_1);

    // Create a segment that lies on the x-axis with extrema [-1.
    // 1]. this will be the reference domain.
    GridGenerator::hyper_cube(triangulation, -1.0, +1.0);

    // Divide the mesh into 2^refinement segnents of equal size.
    triangulation.refine_global(prm_n_refinements);
    print_info();
    // Displace the segment according to the transformations in
    // Geometry. The displacement is given in input using a lambda
    // function that returns the push forward (from chart to space)
    GridTools::transform(
      [this](const Point<dim> &space_point) {
        Point<FunctionOnChart::chart_dim> chart_point{space_point[0]};
        return this->push_forward(chart_point);
      },
      triangulation);
  }
} // namespace lifex::utils