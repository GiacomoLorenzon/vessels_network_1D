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

#ifndef LIFEX_EXAMPLES_VESSEL_1D_HPP_
#define LIFEX_EXAMPLES_VESSEL_1D_HPP_

#include "core/source/core_model.hpp"
#include "core/source/init.hpp"

#include "core/source/numerics/linear_solver_handler.hpp"

#include "examples/vessels_network_1D/function_on_chart.hpp"
#include "examples/vessels_network_1D/mesh_handler_1D.hpp"

#include <deal.II/base/parsed_function.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lifex
{
  /**
   * @brief Class that manages the hemodynamics in a single vessel by
   * projecting N-S equations in 1D. The blood flow model results in a
   * hyperbolic system, which is solved with finite element Taylor-Galerkin
   * numerical scheme. The geometry of the problem is embedded in the physical
   * 3D space. The model's equations follow.
   *
   * \f{eqnarray*}{
   * & \frac{\partial A}{\partial t} + \frac{\partial Q}{\partial z}=0\\
   * & \frac{\partial Q}{\partial t} \frac{\partial}{\partial z}{
   * \left(\frac{\alpha Q^2}{A} \right)} + \frac{A}{\rho}
   * \frac{\partial}{\partial z}{(P - P_{ext})} +K_r \frac{Q}{A} =0
   * \f}
   *
   * where @f$ P - P_{ext} = \beta {(\sqrt{A}-\sqrt{A_0})}/{A_0} @f$. @f$ A @f$
   * and @f$ Q @f$ represent, respectively, the vessel's section area and the
   * mean flux across its section. @f$ \alpha @f$ is the Coriolis coefficient,
   * @f$ \rho @f$ the blood's density, @f$ K_r @f$ the friction coefficient, @f$
   * beta @f$ the stiffness of the vessel's wall and @f$ \psi @f$ is the
   * algebraic relation between the area and the pressure. For further details,
   * please see the references.
   *
   * References:
   * - https://www.mate.polimi.it/biblioteca/add/qmox/mox08.pdf
   */
  class Vessel1D : public CoreModel
  {
  public:
    /**
     * Helper class that wraps boundary data exchanged among vessels.
     */
    struct BoundaryData
    {
      /// Unknown U = [A Q]^T.
      Vector<double> U;
      /// Compatibility conditions.
      Vector<double> CC;
      /// H matrix first left eigenvector.
      Vector<double> L1;
      /// Matrix second left eigenvector.
      Vector<double> L2;
      /// Beta evaluated at the node.
      double beta;
      /// Initial area.
      double A0;
      /// Direction of the vessel.
      Tensor<utils::FunctionOnChart::chart_dim, dim> direction;

      /// Constructor. Allocate with the desired size.
      BoundaryData()
        : U(2)
        , CC(2)
        , L1(2)
        , L2(2){};
    };

    /// Enumeration of the available extremal point types.
    enum class ExtremalPoint
    {
      /// Inflow point.
      Inflow,

      /// Outflow point.
      Outflow
    };

    /// Enumeration of the available branching conditions types.
    enum class BoundaryCondition
    {
      /// Inflow function, given in input.
      Inflow,

      /// Value computed from solving balance equations at the branching nodes,
      BranchingNode,

      /// Type imposed on the outflow points, i.e. the leaf of the graph.
      NonReflecting
    };

    /// @brief Constructor.
    ///
    /// The default boundary conditions are set to
    /// @ref BoundaryCondition::Inflow for the inflow point and to
    /// @ref BoundaryCondition::NonReflecting for the outflow point.
    /// @param[in] input_file file name containing vessel's specific parameters.
    /// @param[in] vessel_id_ id of the vessel.
    /// @param[in] time_step_ uniform time step of the simulation.
    /// @param[in] time_init_ initial time.
    /// @param[in] time_final_ final time.
    /// @param[in] fe_degree_ finite element polynomial degree.
    /// @param[in] alpha_ Coriolis coefficient in the whole network.
    /// @param[in] density_ density of the blood in the whole network.
    /// @param[in] K_r_ friction of the blood in the whole network.
    /// @param[in] inflow_type_ choose whether to impose Area or Flux.
    /// @param[in] enable_output_ save the output files.
    /// @param[in] save_every_n_time_steps_ save the output files every n
    /// time_steps.
    Vessel1D(const std::string & input_file,
             const unsigned int &vessel_id_,
             const double &      time_step_,
             const double &      time_init_,
             const double &      time_final_,
             const unsigned int &fe_degree_,
             const double &      alpha_,
             const double &      density_,
             const double &      K_r_,
             const std::string & inflow_type_,
             const bool &        enable_output_,
             const unsigned int &save_every_n_time_steps_);

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// @brief Solve the problem at fixed time.
    ///
    /// Firstly the current time is updated, then the linear system is
    /// assembled. After checking the CFL condition,
    /// @ref Vessel1D::solve() is called. Lastly, the results are
    /// saved in VTU format.
    void
    solve_time_step();

    /// Setup the system by initializing the finite element space,
    /// reserving memory for variables, initializing sparsity patterns and
    /// interpolating initial or boundary conditions on the nodes.
    void
    setup_system();

    /// @name Methods for interfacing with @ref VesselsNetwork class.
    /// @{

    /// Set physical coordinates of the mesh extremal points and
    /// initialize domain components accordingly. Set also the inflow (outflow)
    /// point identifier.
    /// @param[in] inflow_point [identifier of the vessel extreme within the
    /// network, inflow boundary point coordinates].
    /// @param[in] outflow_point [identifier of the vessel extreme within the
    /// network, outflow boundary point coordinates].
    void
    set_extremal_points(
      const std::pair<const unsigned int, const Point<dim>> &inflow_point,
      const std::pair<const unsigned int, const Point<dim>> &outflow_point);

    /// Get the inflow (outflow) point identifier.
    /// @param[in] point_type Point type chosen. It can be equal to
    /// either @ref ExtremalPoint::Inflow or @ref ExtremalPoint::Inflow.
    /// Otherwise, an exception is thrown.
    /// @return Const reference to the extreme point id chosen.
    const unsigned int &
    get_id_extremal_point(const ExtremalPoint &point_type) const;

    /// Get the inflow (outflow) point in the physical space.
    /// @param[in] point_type Point type chosen. It can be equal to
    /// either @ref ExtremalPoint::Inflow or @ref ExtremalPoint::Inflow.
    /// Otherwise, an exception is thrown.
    /// @return Physical point.
    const Point<dim> &
    get_extremal_point(const ExtremalPoint &point_type) const;

    /// @brief Get the useful quantities contained in BoundaryData struct
    /// at the desired node.
    ///
    /// First get node values for the area and the flux, then retrieve gradients
    /// and finally compute the other quantities.
    /// @param[in] is_ingoing true if one wants to consider the outflow node,
    /// namely the one towards the flux is going into the node.
    /// @return data struct containing pieces of information at the boundary.
    /// Please see BoundaryData documentation.
    Vessel1D::BoundaryData
    get_boundary_data(const bool &is_ingoing) const;

    /// Set the values (area, flux) at the boundary.
    /// @param[in] u pair representing (area, flux) at the boundary point.
    /// @param point_type choose the extremal point type.
    void
    set_boundary_values(const std::pair<double, double> &u,
                        const ExtremalPoint &            point_type);

    /// Set the value of the inflow function evaluated at the current time at
    /// the source vessel inflow node's coordinates.
    void
    set_inflow_source_value(const double &u);

    /// @brief Trigger the boundary condition type which allows to
    /// impose branching boundary conditions at the node in inflow or outflow.
    ///
    /// If the parameter given in input is set to true, namely, the vessel is
    /// ingoing the branching node, @ref bc_outflow_type is set to
    /// @ref BoundaryCondition::BranchingNode.
    /// @param[in] is_ingoing true if the vessel is ingoing the node.
    void
    trigger_branching_bc(const bool &is_ingoing);

    /// @}

    /// Print the state of the vessel, useful while debugging.
    void
    print_vessel_state() const;

    /// Print useful pieces of information regarding the triangulation.
    void
    print_triangulation_info() const;

  protected:
    /// Vessel ID.
    unsigned int vessel_id;

    /// Inflow point ID.
    unsigned int id_extremal_point_0;
    /// Outflow point ID.
    unsigned int id_extremal_point_1;

    /// Vessel parameter name file containing parameters.
    std::string vessel_parameter_file;

    /// This class enumerates degrees of freedom on all vertices, edges, faces,
    /// and cells of the given triangulation. It provides a basis for a FE
    /// space.
    DoFHandler<utils::FunctionOnChart::chart_dim, dim> dof_handler;
    /// Scalar Lagrange FE (within the space of continuous,
    /// piecewise polynomials of degree p in each coordinate direction).
    std::unique_ptr<FE_Q<utils::FunctionOnChart::chart_dim, dim>> fe;
    /// Quadrature formula.
    std::unique_ptr<QGauss<utils::FunctionOnChart::chart_dim>>
      quadrature_formula;
    /// FE space.
    std::unique_ptr<FEValues<utils::FunctionOnChart::chart_dim, dim>> fe_values;
    /// Quadrature face formula.
    std::unique_ptr<QGauss<utils::FunctionOnChart::chart_dim - 1>>
      face_quadrature_formula;
    /// FE face space.
    std::unique_ptr<FEFaceValues<utils::FunctionOnChart::chart_dim, dim>>
      fe_face_values;

    /// Object containing the domain, its length, its direction, its extrema and
    /// pieces of information regarding the chart manifold, namely the mappings
    /// between the physical space and the parametric one.
    utils::MeshHandler1D domain;

    /// System matrices. Area.
    SparseMatrix<double> system_matrix_area;
    /// System matrices. Flux.
    SparseMatrix<double> system_matrix_flux;
    /// Sparsity pattern.
    SparsityPattern sparsity_pattern;
    /// Unknowns at the current time step. Area.
    Vector<double> solution_area; ///< A(t) [cm^2].
    /// Unknowns at the current time step. Flux.
    Vector<double> solution_flux; ///< Q(t) [cm^3 s^-1].
    /// Unknowns at the previous time step. Area.
    Vector<double> old_solution_area; ///< A(t-dt) [cm^2].
    /// Unknowns at the previous time step. Flux.
    Vector<double> old_solution_flux; ///< Q(t-dt) [cm^3 s^-1].
    /// Right hand sides. Area.
    Vector<double> system_rhs_area;
    /// Right hand sides. Flux.
    Vector<double> system_rhs_flux;
    /// Linear solver handler.
    utils::LinearSolverHandler<Vector<double>> solver_vessel_1D;

    /// @name Parameters read from file.
    /// @{

    /// Pointer to beta as function parsed on the 1D space.
    std::shared_ptr<
      Functions::ParsedFunction<utils::FunctionOnChart::chart_dim>>
      prm_function_on_chart_beta;
    /// Object representing beta on chart.
    utils::ParsedFunctionOnChart function_beta;
    /// Node interpolation of beta, i.e. the vessel wall stiffness.
    Vector<double> beta_nodal; ///< [g s^-2].

    /// Pointer to A0 as function parsed on the 1D space.
    std::shared_ptr<
      Functions::ParsedFunction<utils::FunctionOnChart::chart_dim>>
      prm_function_on_chart_A0;
    /// Object representing A0 on chart.
    utils::ParsedFunctionOnChart function_A0;
    /// Node interpolation of the state of the area with P = P_ext.
    Vector<double> A0_nodal; ///< A_0(z) = A(z,t=0) [cm^2].

    /// Adapt the mesh to the expression of beta, to get higher refinement in
    /// correspondence of steep slopes of beta.
    bool prm_adapt_mesh_to_beta;
    /// Fraction of the cell identified by the Kelly estimator that will be
    /// actually refined.
    double prm_top_fraction;
    /// Number of maximum cell to refine at each iteration if the mesh is chosen
    /// to be adapted with the expression of beta.
    unsigned int prm_max_n_cells;
    /// Minimum length of an interval allowed within the mesh when refining
    /// according to beta.
    double prm_minimum_space_amplitude;

    /// Initial condition of the area profile imposed in the reference domain.
    std::shared_ptr<
      Functions::ParsedFunction<utils::FunctionOnChart::chart_dim>>
      prm_ic_on_chart_area;
    /// Object representing the initial area condition on chart.
    utils::ParsedFunctionOnChart ic_area;

    /// Initial condition of the flux.
    std::shared_ptr<
      Functions::ParsedFunction<utils::FunctionOnChart::chart_dim>>
      prm_ic_on_chart_flux;
    /// Object representing the initial flux condition on chart.
    utils::ParsedFunctionOnChart ic_flux;

    /// @}

    /// @name Parameters inherited by the network
    /// @{

    /// @name Blood flow model
    /// @{

    /// Alpha, where @f$ \alpha = \frac{\int_\mathcal{S}s^2\,d\sigma}{A} @f$,
    /// and @f$ s @f$ is the profile law chosen.
    double alpha; ///< [1]
    /// Density of the blood over the whole network. It overwrites the density
    /// if specified in each single vessel.
    double density; ///< [g cm^-3].
    /// Friction. It depends on viscosity.
    double k_r; ///< [cm^2 s^-1].
    /// Inflow condition imposed at the inflow. Available options are "Area",
    /// "Flux". The defaul value is "Area". It is equal to "Area" or
    /// "Flux" only if the vessel is the source one of the network.
    std::string source_inflow_type = "Area";

    /// @}

    /// Time parameters [s].
    double time;
    /// Discrete time step.
    double time_step;
    /// Initial time of the simulation.
    double time_init;
    /// Final time of the simulation.
    double time_final;
    /// Current time step number.
    unsigned int time_step_number;

    /// Finite elements polynomial degree.
    unsigned int fe_degree;

    /// Enable the output saving.
    bool enable_output;
    /// Save every n time_steps the output.
    unsigned int save_every_n_time_steps;

    /// @}

    /// Inflow area value.
    double boundary_inflow_area;
    /// Inflow flux value.
    double boundary_inflow_flux;

    /// Outlow area value.
    double boundary_outflow_area;
    /// Outlow flux value.
    double boundary_outflow_flux;

    /// If the vessel is within a network, Dirichlet conditions at the
    /// boundary nodes are automatically triggered on branching sites.
    /// Inflow condition type.
    BoundaryCondition bc_inflow_type;
    /// Outlow condition type.
    BoundaryCondition bc_outflow_type;

    /// If the vessel is the source one, the value of the boundary condition
    /// imposed within the network is stored in this variable.
    double inflow_source_value;

    /// @brief Set the domain and create the triangulation.
    ///
    /// If the mesh is asked to be adapted to the profile of @f$ \beta @f$,
    /// namely @ref prm_adapt_mesh_to_beta is set to true, then calls the
    /// function @ref adaptive_mesh_refinement. Otherwise, the triangulation is
    /// uniformly spaced.
    void
    setup_domain();

    /// @brief this method adapts the mesh to the profile of @f$ \beta @f$ , so
    /// as to have a better refinement in correspondence of the steepest
    /// sections of the function.
    ///
    /// It starts from the mesh uniformly spaced and it improves the refinement
    /// where necessary. It does not coarsen the mesh elsewhere.
    /// The refinement is done until a maximum number of nodes is obtained or
    /// until a minimum interval diameter is reached. If these conditions are
    /// already fulfilled the triangulation is not modified. Intervals are
    /// selected through dealii::KellyErrorEstimator::estimate() method and a
    /// fraction of them is later flagged and thus refined. The idea is to
    /// estimate the error between the linear interpolation of the function beta
    /// and beta itself on each interval. This is done by integration of the
    /// jump of the gradient of beta between adjacent intervals. For further
    /// details please see the dealii documentation.
    void
    adaptive_mesh_refinement();

    /// Assemble system at the current time step.
    void
    assemble_system();

    /// @brief Impose the boundary conditions.
    ///
    /// The method impose the proper boundary coditions at the extremal nodes
    /// of the vessel. They are set according to the label of the extrema
    /// specified by @ref Vessel1D::BoundaryCondition.
    void
    impose_boundary_conditions();

    /// Solve linear system using the Conjugate Gradient method.
    void
    solve();

    /// @brief Save current area and flux on file.
    ///
    /// The file is named  "solutionZZZ-YYYYYY.vtu". "ZZZ" stands for the vessel
    /// ID and "YYYYYY" is a sequentially increasing integer. It is located in
    /// @ref Core::prm_output_directory.
    void
    output_results() const;

    /// Check the Courant–Friedrichs–Lewy condition, i.e.
    /// @f$ dt < \frac{\sqrt(3)}{3} \frac{h}{\lambda_{MAX}} @f$.
    /// @return true if the CFL condition is satisfied, false otherwise.
    bool
    check_CFL() const;
  };

} // namespace lifex

#endif // LIFEX_EXAMPLES_VESSEL_1D_HPP_
