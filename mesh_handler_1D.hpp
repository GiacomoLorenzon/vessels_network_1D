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

#ifndef LIFEX_MESH_HANDLER_1D
#define LIFEX_MESH_HANDLER_1D

#include "core/source/core_model.hpp"

#include "examples/vessels_network_1D/function_on_chart.hpp"

#include <deal.II/base/parsed_function.h>

#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lifex::utils
{
  /**
   * @brief Mesh defined on a one-dimensional manifold embedded in a
   * @f$d@f$-dimensional space.
   *
   * Let @f$\Omega \subset \mathbb{R}@f$ be the reference domain and let @f$
   * \mathbf{r} : \Omega \rightarrow \mathbb{R}^{d} @f$ be the parametrization
   * of the curve @f$ \Gamma \subset \mathbb{R}^{d} @f$. This class allows to
   * handle the domain @f$ \Gamma @f$ and its partition.
   * It also provides two core methods: @ref push_forward and @ref pull_back.
   * Those two methods encode the relation between the domain in the physical
   * space @f$ \Gamma @f$ and the domain on chart @f$ \Omega @f$.
   */
  class MeshHandler1D : public ChartManifold<FunctionOnChart::chart_dim,
                                             dim,
                                             FunctionOnChart::chart_dim>,
                        public CoreModel
  {
  public:
    /// Enumeration of the available geometry types.
    enum class GeometryType
    {
      /// Generate a segment in space.
      ///
      /// The segment can be generated either by linking two given vertices in
      /// the 3D space or by computing the position of the second vertex, given
      /// the first one together with the segment's length and direction.
      ///
      /// In a transport problem, boundary IDs are numbered as follows:
      /// - 0 is assigned to the inflow point;
      /// - 1 is assigned to the outflow point.
      Linear,

      /// Mesh generated using user-defined methods.
      File
    };

    /// Constructor.
    ///
    /// Construct the segment in the 3D space, given the string of the
    /// subsection from which to read input data.
    MeshHandler1D(const std::string &subsection,
                  const bool &       parse_mesh_parameters = true);

    /// Assignment operator deleted.
    MeshHandler1D &
    operator=(const MeshHandler1D &other) = delete;

    /// Destructor.
    ~MeshHandler1D() = default;

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// Initialize the segment in the 3D space, given the two extremal points.
    /// @param[in] extremal_space_point_0 inflow point;
    /// @param[in] extremal_space_point_1 outflow point.
    void
    initialize_linear_segment(const Point<dim> &extremal_space_point_0,
                              const Point<dim> &extremal_space_point_1);

    /// Initialize the segment in the 3D space, given one extremal point, the
    /// length and the direction of the segment.
    /// @param[in] extremal_space_point_0 inflow point;
    /// @param[in] length_ length of the segment;
    /// @param[in] direction_ direction of the segment, not necessarily
    /// normalized.
    void
    initialize_linear_segment(
      const Point<dim> &                             extremal_space_point_0,
      double                                         length_,
      const Tensor<FunctionOnChart::chart_dim, dim> &direction_);

    /// @name Mapping between parametric and physical space.
    /// @{

    /// @brief Map from the physical space to the 1D parametrized space
    /// @f$ F^{-1} : \mathbb{R}^3 \rightarrow [-1, 1] @f$ overridden from
    /// dealii::ChartManifold.
    ///
    /// After performing the mapping, the point found is projected onto the
    /// parametrized space. In other words, the point is constrained to belong
    /// to the reference domain.
    /// @param[in] space_point point in the physical space.
    /// @return point mapped into the parametric space.
    virtual Point<FunctionOnChart::chart_dim>
    pull_back(const Point<dim> &space_point) const override;

    /// @brief Map from the 1D parametric space to the physical one
    /// @f$ F : [-1, 1] \rightarrow \mathbb{R}^3 @f$ overridden from
    /// dealii::ChartManifold.
    ///
    /// If the point given in input does not lie in the reference domain, it is
    /// projected onto it.
    /// @param[in] chart_point point in the parametric space.
    /// @return point mapped into the physical space.
    virtual Point<dim>
    push_forward(
      const Point<FunctionOnChart::chart_dim> &chart_point) const override;

    /// @}

    /// Override of Manifold::clone. This class is prevented from being
    /// cloneable. Indeed, the instance of @ref triangulation would be ill
    /// formed. If one tries to clone it nonetheless, an exception is thrown.
    virtual std::unique_ptr<Manifold<FunctionOnChart::chart_dim, dim>>
    clone() const override;

    /// Compute the directional derivative of a function @f$ f @f$
    /// according to the gradient theorem, i.e.
    /// @f$ \frac{\partial f}{\partial \mathbf{v}} =
    /// \nabla f\cdot \mathbf{v} @f$, where @f$ \mathbf{v}@f$ is the versor
    /// of the linear segment.
    /// @param[in] gradient Full gradient @f$\nabla f@f$.
    /// @param[in] space_point default value is @f$ (0,0,0) @f$. One can avoid
    /// passing this argument if the geometry is linear.
    /// @return Derivative computed along the versor of the linear segment.
    double
    project_gradient(const Tensor<FunctionOnChart::chart_dim, dim> &gradient,
                     const Point<dim> &space_point = Point<dim>(0.0,
                                                                0.0,
                                                                0.0)) const;

    /// Print pieces of information about the mesh.
    void
    print_info() const;

    /// Create the mesh.
    void
    create_mesh();

    /// Save the current triangulation to a vtk file.
    void
    triangulation_to_vtk() const;

    /// Get the underlying triangulation object.
    /// @return triangulation of the current mesh.
    Triangulation<FunctionOnChart::chart_dim, dim> &
    get_triangulation();

    /// Get the underlying triangulation object, <kbd>const</kbd> version.
    /// @return triangulation of the current mesh.
    const Triangulation<FunctionOnChart::chart_dim, dim> &
    get_triangulation() const;

    /// Get an object representing the Gaussian quadrature formula on
    /// the triangulation.
    /// @param[in] n_points number of points for the quadrature formula chosen.
    /// @return unique pointer to dealii::QGauss<FunctionOnChart::chart_dim>
    /// object.
    std::unique_ptr<QGauss<FunctionOnChart::chart_dim>>
    get_quadrature_gauss(const unsigned int &n_points) const;

    /// Get an object representing the Lagrange finite element space on
    /// the triangulation.
    /// @param[in] degree degree of the FE chosen.
    /// @return unique pointer to dealii::FE_Q<FunctionOnChart::chart_dim, dim>
    /// object.
    std::unique_ptr<FE_Q<FunctionOnChart::chart_dim, dim>>
    get_fe_lagrange(const unsigned int &degree) const;

    /// Get the length of the segment if initialized.
    double
    get_length() const;

    /// Get the number of uniform refinements.
    double
    get_n_refinements() const;

    /// Get the direction of the segment in the physical space.
    /// @return versor of a vector representing the segment direction.
    Tensor<FunctionOnChart::chart_dim, dim>
    get_direction() const;

    /// Get the physical coordinates in the space of one of the
    /// extremal points.
    /// @param[in] point_index Available options are 0 and 1.
    /// @return Physical point selected.
    const Point<dim> &
    get_extremal_point(const unsigned int &point_index) const;

    /// @brief Flag intervals for local refinement, according to the criterion
    /// given in input.
    ///
    /// The criterion here is represented by a vector of floats,
    /// which can be filled by other methods, such as
    /// dealii::KellyErrorEstimator::estimate(). The top fraction of intervals
    /// with the correspondent's highest value will be flagged for later
    /// refinement.
    /// @param[in] criterion vector of float numbers with the same length of the
    /// number of intervals of the triangulation containing the values according
    /// to which the refinement is performed.
    /// @param[in] top_fraction fraction of the intervals satisfying the
    /// criterion that are actually refined.
    /// @param[in] max_n_cells maximum number of intervals allowed within the
    /// triangulation.
    void
    flag_intervals_for_refinement(const Vector<float> &criterion,
                                  const double &       top_fraction,
                                  const unsigned int & max_n_cells);

  private:
    /// Distinguish linear geometry from other kind of geometries.
    GeometryType prm_geometry_type;
    /// Mesh file name used to read a non-linear geometry type.
    std::string prm_mesh_file_name;
    /// Parse mesh extreme points or file name with @ref parse_parameters. It
    /// might be set to false if the mesh is complex and it is read from another
    /// class. In this case the data members of this class should be initalized
    /// properly.
    bool prm_parse_mesh_parameters = false;
    /// Triangulation of the domain.
    Triangulation<FunctionOnChart::chart_dim, dim> triangulation;
    /// Number of uniform refinements of the mesh.
    unsigned int prm_n_refinements;

    /// @name Line geometry data.
    /// @{

    /// First extremal point.
    Point<dim> extremal_point_0;
    /// Second extremal point.
    Point<dim> extremal_point_1;
    /// Segment direction, normalized.
    Tensor<FunctionOnChart::chart_dim, dim> direction;
    /// Segment length. Default is -1.0.
    double length = -1.0;

    /// @}

    /// Set the flags on the intervals that will be refined.
    /// @param[in] criterion vector of float numbers with the same length of the
    /// number of intervals of the triangulation containing the values according
    /// to which the refinement is performed. This method is used within
    /// flag_intervals_for_refinement() which is the one exposed for the user.
    /// @param[in] threshold value over which the interval is flagged.
    /// @param[in] max_to_mark maximum number of intervals that can be flagged
    /// to further refinement.
    void
    set_intervals_refinement_flag(const Vector<float> &criterion,
                                  double               threshold,
                                  const unsigned int & max_to_mark);

  protected:
    /// Linear version of the @ref pull_back function.
    Point<FunctionOnChart::chart_dim>
    linear_pull_back(const Point<dim> &space_point) const;

    /// Linear version of the @ref push_forward function.
    Point<dim>
    linear_push_forward(
      const Point<FunctionOnChart::chart_dim> &chart_point) const;

    /// Linear version of the @ref project_gradient function.
    double
    linear_project_gradient(
      const Tensor<FunctionOnChart::chart_dim, dim> &gradient) const;

    /// Linear version of the @ref create_mesh function.
    void
    create_linear_mesh();
  };
} // namespace lifex::utils

#endif // LIFEX_MESH_HANDLER_1D
