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

#include "examples/vessels_network_1D/vessel_1D.hpp"

#include <deal.II/dofs/dof_renumbering.h>

#include <algorithm>

namespace lifex
{
  Vessel1D::Vessel1D(const std::string & input_file,
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
                     const unsigned int &save_every_n_time_steps_)
    : CoreModel("Vessel 1D")
    , vessel_id(vessel_id_)
    , vessel_parameter_file(input_file)
    , domain("Mesh and space discretization", false)
    , solver_vessel_1D("Linear solver", {"CG", "GMRES"}, "CG")
    , function_beta(domain)
    , function_A0(domain)
    , ic_area(domain)
    , ic_flux(domain)
    , alpha(alpha_)
    , density(density_)
    , k_r(K_r_)
    , source_inflow_type(inflow_type_)
    , time_step(time_step_)
    , time_init(time_init_)
    , time_final(time_final_)
    , fe_degree(fe_degree_)
    , enable_output(enable_output_)
    , save_every_n_time_steps(save_every_n_time_steps_)

  {
    bc_inflow_type  = BoundaryCondition::Inflow;
    bc_outflow_type = BoundaryCondition::NonReflecting;
  }

  void
  Vessel1D::declare_parameters(ParamHandler &params) const
  {
    params.enter_subsection("Physical constants and models");
    {
      params.enter_subsection("Beta");
      {
        // Mathematical expression of the wall stiffness. The function is
        // defined on the reference domain [-1, 1].
        Functions::ParsedFunction<
          utils::FunctionOnChart::chart_dim>::declare_parameters(params);
      }
      params.leave_subsection();
      params.enter_subsection("A0");
      {
        // Mathematical expression of the area profile when the pressure is
        // equal to the external pressure. The function is defined on the
        // reference domain [-1, 1].
        Functions::ParsedFunction<
          utils::FunctionOnChart::chart_dim>::declare_parameters(params);
      }
      params.leave_subsection();
    }
    params.leave_subsection();

    domain.declare_parameters(params);

    params.enter_subsection("Triangulation");
    {
      params.declare_entry("Adapt the mesh to the stiffness",
                           "false",
                           Patterns::Bool(),
                           "Choose if you want to adapt the mesh refinement to "
                           "the stiffness, getting it more refined where its "
                           "derivative is more steep.");
      params.declare_entry("Fraction of cells to refine",
                           "0.30",
                           Patterns::Double(0.0),
                           "Fraction of the cell identified by the Kelly "
                           "estimator that will be actually refined.");
      params.declare_entry("Maximum number of cells",
                           "1024",
                           Patterns::Integer(0),
                           "Maximum number of intervals flagged at each "
                           "iteration of the refinement algorithm.");
      params.declare_entry(
        "Minimum mesh size",
        "0.01",
        Patterns::Double(0.0),
        "Minimuim size of the mesh element after refinement.");
    }
    params.leave_subsection();

    // Initial condition area subsection.
    params.enter_subsection("Initial condition area");
    {
      // Mathematical expression of the initial condition of the area. The
      // function is defined on the reference domain [-1, 1]."
      Functions::ParsedFunction<
        utils::FunctionOnChart::chart_dim>::declare_parameters(params);
    }
    params.leave_subsection();

    // Initial condition flux subsection.
    params.enter_subsection("Initial condition flux");
    {
      // Mathematical expression of the initial condition of the flux. The
      // function is defined on the reference domain [-1, 1]."
      Functions::ParsedFunction<
        utils::FunctionOnChart::chart_dim>::declare_parameters(params);
    }
    params.leave_subsection();

    solver_vessel_1D.declare_parameters(params);
  }

  void
  Vessel1D::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse_input(vessel_parameter_file);

    params.enter_subsection("Physical constants and models");
    {
      params.enter_subsection("Beta");
      {
        prm_function_on_chart_beta = std::make_shared<
          Functions::ParsedFunction<utils::FunctionOnChart::chart_dim>>();
        prm_function_on_chart_beta->parse_parameters(params);
      }
      params.leave_subsection();
      params.enter_subsection("A0");
      {
        prm_function_on_chart_A0 = std::make_shared<
          Functions::ParsedFunction<utils::FunctionOnChart::chart_dim>>();
        prm_function_on_chart_A0->parse_parameters(params);
      }
      params.leave_subsection();
    }
    params.leave_subsection();

    // Read mesh type.
    domain.parse_parameters(params);

    params.enter_subsection("Triangulation");
    {
      prm_adapt_mesh_to_beta =
        params.get_bool("Adapt the mesh to the stiffness");
      prm_top_fraction = params.get_double("Fraction of cells to refine");
      prm_max_n_cells  = params.get_integer("Maximum number of cells");
      prm_minimum_space_amplitude = params.get_double("Minimum mesh size");
    }
    params.leave_subsection();

    // Read initial condition area.
    params.enter_subsection("Initial condition area");
    {
      prm_ic_on_chart_area = std::make_shared<
        Functions::ParsedFunction<utils::FunctionOnChart::chart_dim>>();
      prm_ic_on_chart_area->parse_parameters(params);
    }
    params.leave_subsection();

    // Read initial condition flux.
    params.enter_subsection("Initial condition flux");
    {
      prm_ic_on_chart_flux = std::make_shared<
        Functions::ParsedFunction<utils::FunctionOnChart::chart_dim>>();
      prm_ic_on_chart_flux->parse_parameters(params);
    }
    params.leave_subsection();

    solver_vessel_1D.parse_parameters(params);
  }

  void
  Vessel1D::solve_time_step()
  {
    // Advance in time.
    time += time_step;
    ++time_step_number;

    // At time fixed:
    //  - update the solutions.
    old_solution_area = solution_area;
    old_solution_flux = solution_flux;

    // - assemble system and set border conditions.
    assemble_system();

    // - check CFL condition.
    Assert(check_CFL(),
           ExcMessage("CFL condition [time_step > std::sqrt(3) / 3 * h / "
                      "max_lambda] not satisfied by vessel " +
                      std::to_string(vessel_id) + ".\n"));

    // - solve the system.
    solve();

    // - print the results.
    output_results();
  }

  void
  Vessel1D::set_extremal_points(
    const std::pair<const unsigned int, const Point<dim>> &inflow_point,
    const std::pair<const unsigned int, const Point<dim>> &outflow_point)
  {
    // Inflow point.
    id_extremal_point_0 = std::get<0>(inflow_point);

    // Outflow point.
    id_extremal_point_1 = std::get<0>(outflow_point);

    this->domain.initialize_linear_segment(std::get<1>(inflow_point),
                                           std::get<1>(outflow_point));
  }

  const unsigned int &
  Vessel1D::get_id_extremal_point(const ExtremalPoint &point_type) const
  {
    Assert(point_type == ExtremalPoint::Inflow ||
             point_type == ExtremalPoint::Outflow,
           ExcInvalidState());

    if (point_type == ExtremalPoint::Inflow)
      return id_extremal_point_0;
    else // if (point_type == ExtremalPoint::Outflow)
      return id_extremal_point_1;
  }

  const Point<dim> &
  Vessel1D::get_extremal_point(const ExtremalPoint &point_type) const
  {
    Assert(point_type == ExtremalPoint::Inflow ||
             point_type == ExtremalPoint::Outflow,
           ExcInvalidState());

    if (point_type == ExtremalPoint::Inflow)
      return domain.get_extremal_point(0);
    else // if (point_type == ExtremalPoint::Outflow)
      return domain.get_extremal_point(1);
  }

  Vessel1D::BoundaryData
  Vessel1D::get_boundary_data(const bool &is_ingoing) const
  {
    const unsigned int dofs_per_face = 1;

    double dA_dz = 0.0, dQ_dz = 0.0, dA0_dz = 0.0, dbeta_dz = 0.0;
    double A0 = 0.0, beta = 0.0, A = 0.0, Q = 0.0;

    /// Useful values and retrieve gradients.
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_area(dofs_per_face);
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_flux(dofs_per_face);
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_A0(dofs_per_face);
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_beta(dofs_per_face);

    std::vector<double> loc_A0(dofs_per_face);
    std::vector<double> loc_beta(dofs_per_face);
    std::vector<double> loc_solution_area(dofs_per_face);
    std::vector<double> loc_solution_flux(dofs_per_face);

    // If the vessel is ingoing into the node, we must extract its outlet
    // values. Notice that since it's a segment, there are just two extrema.
    for (auto interval : filter_iterators(dof_handler.active_cell_iterators(),
                                          IteratorFilters::AtBoundary()))
      for (unsigned int face_id = 0; face_id < interval->n_faces(); ++face_id)
        if (interval->face(face_id)->at_boundary())
          {
            const types::boundary_id face_bd_id =
              interval->face(face_id)->boundary_id();

            fe_face_values->reinit(interval, face_id);

            fe_face_values->get_function_values(A0_nodal, loc_A0);
            fe_face_values->get_function_values(beta_nodal, loc_beta);
            fe_face_values->get_function_values(old_solution_area,
                                                loc_solution_area);
            fe_face_values->get_function_values(old_solution_flux,
                                                loc_solution_flux);

            fe_face_values->get_function_gradients(A0_nodal, loc_grad_A0);
            fe_face_values->get_function_gradients(beta_nodal, loc_grad_beta);
            fe_face_values->get_function_gradients(old_solution_area,
                                                   loc_grad_area);
            fe_face_values->get_function_gradients(old_solution_flux,
                                                   loc_grad_flux);

            if (face_bd_id != is_ingoing)
              {
                A0   = loc_A0.front();
                beta = loc_beta.front();
                A    = loc_solution_area.front();
                Q    = loc_solution_flux.front();

                dA0_dz   = domain.project_gradient(loc_grad_A0.front());
                dbeta_dz = domain.project_gradient(loc_grad_beta.front());
                dA_dz    = domain.project_gradient(loc_grad_area.front());
                dQ_dz    = domain.project_gradient(loc_grad_flux.front());
              }
            else if (face_bd_id == is_ingoing)
              {
                A0   = loc_A0.back();
                beta = loc_beta.back();
                A    = loc_solution_area.back();
                Q    = loc_solution_flux.back();

                dA0_dz   = domain.project_gradient(loc_grad_A0.back());
                dbeta_dz = domain.project_gradient(loc_grad_beta.back());
                dA_dz    = domain.project_gradient(loc_grad_area.back());
                dQ_dz    = domain.project_gradient(loc_grad_flux.back());
              }
          }

    FullMatrix<double> H(2, 2);
    Vector<double>     dU_dz(2);
    Vector<double>     B(2);
    double             c_alpha;

    // psi = beta / A0 * (A^(1/2) - A0^(1/2)).
    const double d_psi_dA0 =
      -beta * ((std::sqrt(A) - std::sqrt(A0)) / std::pow(A0, 2.0) +
               1.0 / (2.0 * std::pow(A0, 1.5)));
    const double d_psi_dA    = beta / (2.0 * std::sqrt(A) * A0);
    const double d_psi_dbeta = (std::sqrt(A) - std::sqrt(A0)) / A0;

    Vessel1D::BoundaryData data;

    // Matrices evaluated at the boundary_values.
    // U = [A, Q]^T;
    data.U(0) = A;
    data.U(1) = Q;

    // H.
    H(0, 0) = 0.0;
    H(0, 1) = 1.0;
    H(1, 0) = (A / density) * d_psi_dA - alpha * std::pow((Q / A), 2.0);
    H(1, 1) = 2.0 * alpha * Q / A;

    // dU_dz.
    dU_dz(0) = dA_dz;
    dU_dz(1) = dQ_dz;

    // B.
    B(0) = 0.0;
    B(1) = k_r * Q / A +
           (A / density) * (d_psi_dA0 * dA0_dz + d_psi_dbeta * dbeta_dz);

    // Compatibility conditions.
    data.CC(0) =
      data.U[0] - time_step * (H(0, 0) * dU_dz(0) + H(0, 1) * dU_dz(1) + B[0]);
    data.CC(1) =
      data.U[1] - time_step * (H(1, 0) * dU_dz(0) + H(1, 1) * dU_dz(1) + B[1]);

    // c_alpha.
    c_alpha = std::sqrt(beta / (2.0 * density * A0) * std::sqrt(A) +
                        std::pow((Q / A), 2.0) * alpha * (alpha - 1.0));

    // H matrix left eigenvectors.
    // First:
    data.L1(0) = c_alpha - alpha * Q / A;
    data.L1(1) = 1.0;
    // Second:
    data.L2(0) = -c_alpha - alpha * Q / A;
    data.L2(1) = 1.0;

    data.beta      = beta;
    data.A0        = A0;
    data.direction = domain.get_direction();

    return data;
  }

  void
  Vessel1D::set_boundary_values(const std::pair<double, double> &u,
                                const ExtremalPoint &            point_type)
  {
    Assert(point_type == ExtremalPoint::Inflow ||
             point_type == ExtremalPoint::Outflow,
           ExcInvalidState());

    if (point_type == ExtremalPoint::Outflow)
      {
        boundary_outflow_area = u.first;
        boundary_outflow_flux = u.second;
      }
    else // (point_type == ExtremalPoint::Intflow)
      {
        boundary_inflow_area = u.first;
        boundary_inflow_flux = u.second;
      }
  }

  void
  Vessel1D::set_inflow_source_value(const double &u)
  {
    inflow_source_value = u;
  }

  void
  Vessel1D::trigger_branching_bc(const bool &is_ingoing)
  {
    if (is_ingoing)
      {
        bc_outflow_type = BoundaryCondition::BranchingNode;
      }
    else // is_outgoing
      {
        bc_inflow_type = BoundaryCondition::BranchingNode;
      }
  }

  void
  Vessel1D::print_vessel_state() const
  {
    pcout << "---------------------------------------------" << std::endl;
    pcout << " vessel ID: " << vessel_id << std::endl;
    pcout << std::endl;

    pcout << " vessel_parameter_file: " << vessel_parameter_file << std::endl;
    pcout << std::endl;

    pcout << " time          : " << time << std::endl;
    pcout << " time_step     : " << time_step << std::endl;
    pcout << " time_init     : " << time_init << std::endl;
    pcout << " time_final    : " << time_final << std::endl;
    pcout << std::endl;

    pcout << " source_inflow_type : " << source_inflow_type << std::endl;
    pcout << std::endl;

    pcout << " time_step_number : " << time_step_number << std::endl;
    pcout << std::endl;

    pcout << " enable_output: " << enable_output << std::endl;
    pcout << " save_every_n_time_steps: " << save_every_n_time_steps
          << std::endl;
    pcout << std::endl;

    pcout << " alpha  : " << alpha << std::endl;
    pcout << " density: " << density << std::endl;
    pcout << " k_r    : " << k_r << std::endl;
    pcout << std::endl;

    pcout << " inflow point ID : " << id_extremal_point_0 << std::endl;
    pcout << " inflow point       : "
          << get_extremal_point(ExtremalPoint::Inflow) << std::endl;
    pcout << " outflow point ID: " << id_extremal_point_1 << std::endl;
    pcout << " outflow point      : "
          << get_extremal_point(ExtremalPoint::Outflow) << std::endl;
    pcout << std::endl;

    pcout << " boundary_inflow_area         : " << boundary_inflow_area
          << std::endl;
    pcout << " boundary_inflow_flux         : " << boundary_inflow_flux
          << std::endl;
    pcout << std::endl;

    pcout << " boundary_outflow_area        : " << boundary_outflow_area
          << std::endl;
    pcout << " boundary_outflow_flux        : " << boundary_outflow_flux
          << std::endl;
    pcout << std::endl;
  }

  void
  Vessel1D::print_triangulation_info() const
  {
    pcout << "=============================================" << std::endl
          << " Triangulation " << std::endl
          << "=============================================" << std::endl
          << " Number of active cells:       "
          << domain.get_triangulation().n_active_cells() << std::endl
          << " Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl
          << " FE degree:                    " << fe->degree << std::endl
          << " Quadrature formula degree:    " << fe->degree + 1 << std::endl
          << "=============================================" << std::endl
          << std::endl;
  }

  void
  Vessel1D::setup_domain()
  {
    domain.create_mesh();

    if (prm_adapt_mesh_to_beta)
      adaptive_mesh_refinement();
  }

  void
  Vessel1D::adaptive_mesh_refinement()
  {
    // Compute the minimum length that will be reached if the refinement is
    // executed.
    double h_min =
      domain.get_length() / (2.0 * std::pow(2, domain.get_n_refinements()));

    // Stop condition.
    bool         is_refining = true;
    unsigned int prev_number_of_cells =
      domain.get_triangulation().n_active_cells();

    while (h_min > prm_minimum_space_amplitude && is_refining)
      {
        fe = domain.get_fe_lagrange(1);
        dof_handler.reinit(domain.get_triangulation());
        dof_handler.distribute_dofs(*fe);

        // Interpolate beta.
        function_beta.set_function_on_chart(prm_function_on_chart_beta);
        beta_nodal.reinit(dof_handler.n_dofs());
        VectorTools::interpolate(dof_handler, function_beta, beta_nodal);

        // Prepare the vector in which to save the error for each interval.
        Vector<float> estimated_error_per_cell(prev_number_of_cells);

        // Compute the estimate according to KellyErrorEstimator algorithm.
        KellyErrorEstimator<utils::FunctionOnChart::chart_dim, dim>::estimate(
          dof_handler,
          QGauss<utils::FunctionOnChart::chart_dim - 1>(fe->degree + 1),
          {},
          beta_nodal,
          estimated_error_per_cell);

        // Flag the cells to refine.
        domain.flag_intervals_for_refinement(estimated_error_per_cell,
                                             prm_top_fraction,
                                             prm_max_n_cells);

        // Do the actual refinement.
        Triangulation<utils::FunctionOnChart::chart_dim, dim>
          &ref_triangulation = domain.get_triangulation();
        ref_triangulation.execute_coarsening_and_refinement();

        // Stop condition.
        if (prev_number_of_cells == domain.get_triangulation().n_active_cells())
          is_refining = false;
        else
          prev_number_of_cells = domain.get_triangulation().n_active_cells();

        for (const auto &interval : dof_handler.active_cell_iterators())
          h_min = std::min(h_min, interval->diameter());
      }
  }

  void
  Vessel1D::setup_system()
  {
    // Create the mesh and the maps between the physical space and the
    // reference one.
    setup_domain();

    // Get lagrange finite elements in the space Q.
    fe = domain.get_fe_lagrange(fe_degree);
    // Get quadrature formula.
    quadrature_formula = domain.get_quadrature_gauss(fe->degree + 1);
    // Get face quadrature formula.
    face_quadrature_formula =
      std::make_unique<QGauss<utils::FunctionOnChart::chart_dim - 1>>(
        fe->degree + 1);

    // Setup the mesh and global numbering.
    dof_handler.reinit(domain.get_triangulation());
    // Generate an enumeration of the degrees of freedom.
    dof_handler.distribute_dofs(*fe);

    ic_area.set_function_on_chart(prm_ic_on_chart_area);
    ic_flux.set_function_on_chart(prm_ic_on_chart_flux);

    function_beta.set_function_on_chart(prm_function_on_chart_beta);
    function_A0.set_function_on_chart(prm_function_on_chart_A0);

    // Setup sparsity pattern for later use in the mass matrix.
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix_area.reinit(sparsity_pattern);
    system_matrix_flux.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<utils::FunctionOnChart::chart_dim>(
                                        fe->degree + 1),
                                      system_matrix_area);
    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<utils::FunctionOnChart::chart_dim>(
                                        fe->degree + 1),
                                      system_matrix_flux);

    // Reserve the right amount of memory for speed.
    solution_area.reinit(dof_handler.n_dofs());
    solution_flux.reinit(dof_handler.n_dofs());

    old_solution_area.reinit(dof_handler.n_dofs());
    old_solution_flux.reinit(dof_handler.n_dofs());

    A0_nodal.reinit(dof_handler.n_dofs());
    beta_nodal.reinit(dof_handler.n_dofs());

    system_rhs_area.reinit(dof_handler.n_dofs());
    system_rhs_flux.reinit(dof_handler.n_dofs());

    // Initilize time variables.
    time             = time_init;
    time_step_number = 0;
    // Finite element space.
    fe_values =
      std::make_unique<FEValues<utils::FunctionOnChart::chart_dim, dim>>(
        *fe,
        *quadrature_formula,
        update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

    fe_face_values =
      std::make_unique<FEFaceValues<utils::FunctionOnChart::chart_dim, dim>>(
        *fe,
        *face_quadrature_formula,
        update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

    // Set the initial conditions on the FE space.
    VectorTools::interpolate(dof_handler, ic_area, solution_area);
    VectorTools::interpolate(dof_handler, ic_flux, solution_flux);

    VectorTools::interpolate(dof_handler, function_beta, beta_nodal);
    VectorTools::interpolate(dof_handler, function_A0, A0_nodal);

    old_solution_area = solution_area;
    old_solution_flux = solution_flux;

    output_results();
  }

  void
  Vessel1D::assemble_system()
  {
    // Save local quantities.
    const unsigned int &dofs_per_cell = fe->n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    Vector<double> loc_rhs_area(dofs_per_cell);
    Vector<double> loc_rhs_flux(dofs_per_cell);

    std::vector<double> loc_area(dofs_per_cell);
    std::vector<double> loc_flux(dofs_per_cell);
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_area(dofs_per_cell);
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_flux(dofs_per_cell);

    std::vector<double> loc_A0(dofs_per_cell);
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_A0(dofs_per_cell);

    std::vector<double> loc_beta(dofs_per_cell);
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_beta(dofs_per_cell);

    // Reinitialize rhs at every time-step.
    system_rhs_area = 0;
    system_rhs_flux = 0;

    // Loop over intervals.
    for (const auto &interval : dof_handler.active_cell_iterators())
      {
        fe_values->reinit(interval);

        loc_rhs_area = 0;
        loc_rhs_flux = 0;

        fe_values->get_function_values(old_solution_area, loc_area);
        fe_values->get_function_values(old_solution_flux, loc_flux);
        fe_values->get_function_gradients(old_solution_area, loc_grad_area);
        fe_values->get_function_gradients(old_solution_flux, loc_grad_flux);

        fe_values->get_function_values(beta_nodal, loc_beta);
        fe_values->get_function_gradients(beta_nodal, loc_grad_beta);
        fe_values->get_function_values(A0_nodal, loc_A0);
        fe_values->get_function_gradients(A0_nodal, loc_grad_A0);

        // Loop over quadrature nodes.
        for (const unsigned int q : fe_values->quadrature_point_indices())
          {
            const double &Q    = loc_flux[q];
            const double &A    = loc_area[q];
            const double &A0   = loc_A0[q];
            const double &beta = loc_beta[q];

            const double dQ_dz    = domain.project_gradient(loc_grad_flux[q]);
            const double dA_dz    = domain.project_gradient(loc_grad_area[q]);
            const double dA0_dz   = domain.project_gradient(loc_grad_A0[q]);
            const double dbeta_dz = domain.project_gradient(loc_grad_beta[q]);

            // psi = beta / A0 * (A^(1/2) - A0^(1/2)).
            const double d_psi_dA0 =
              -beta * ((std::sqrt(A) - std::sqrt(A0)) / std::pow(A0, 2.0) +
                       1.0 / (2.0 * std::pow(A0, 1.5)));
            const double d_psi_dA    = beta / (2.0 * std::sqrt(A) * A0);
            const double d_psi_dbeta = (std::sqrt(A) - std::sqrt(A0)) / A0;

            // C1 =  int(c1^2, 0, A).
            const double C1 = beta * std::pow(A, 1.5) / (3.0 * density * A0);

            const double dC1_dz =
              std::sqrt(A) / (density * A0) *
              ((beta / 2.0) * dA_dz - (A * beta) / (3.0 * A0) * dA0_dz +
               (A / 3.0) * dbeta_dz);
            const double dC1_dA0 =
              -std::pow(A, 1.5) * beta / (3.0 * std::pow(A0, 2.0) * density);
            const double dC1_dbeta = std::pow(A, 1.5) / (3.0 * A0 * density);

            const std::vector<std::vector<double>> H(
              {{0.0, 1.0},
               {(A / density) * d_psi_dA - alpha * std::pow((Q / A), 2.0),
                2.0 * alpha * Q / A}});

            const std::vector<double> B(
              {0.0,
               k_r * Q / A + (A / density) *
                               (d_psi_dA0 * dA0_dz + d_psi_dbeta * dbeta_dz)});

            const std::vector<double> F({Q, alpha * std::pow(Q, 2.0) / A + C1});

            const std::vector<double> dF_dz(
              {dQ_dz,
               2.0 * alpha * Q / A * dQ_dz -
                 alpha * (std::pow((Q / A), 2.0)) * dA_dz + dC1_dz});

            const std::vector<double> S(
              {B[0], B[1] - (dC1_dA0 * dA0_dz + dC1_dbeta * dbeta_dz)});

            const std::vector<std::vector<double>> dS_dU(
              {{0.0, 0.0},
               {-k_r * Q / (std::pow(A, 2.0)) -
                  beta / (density * A0) *
                    ((std::sqrt(A) - std::sqrt(A0)) / A0 +
                     1.0 / (2.0 * std::sqrt(A0))) *
                    dA0_dz +
                  (std::sqrt(A) - std::sqrt(A0)) / (A0 * density) * dbeta_dz,
                k_r / A}});

            for (const unsigned int i : fe_values->dof_indices())
              {
                // Shape functions derivatives.
                const double dPhi_i_q =
                  domain.project_gradient(fe_values->shape_grad(i, q));

                // Local assembly.
                // dt*(F-0.5*dt*H*S,dPhi_dz).
                loc_rhs_area(i) +=
                  time_step *
                  (F[0] - time_step / 2.0 * (H[0][0] * S[0] + H[0][1] * S[1])) *
                  dPhi_i_q * fe_values->JxW(q);
                loc_rhs_flux(i) +=
                  time_step *
                  (F[1] - time_step / 2.0 * (H[1][0] * S[0] + H[1][1] * S[1])) *
                  dPhi_i_q * fe_values->JxW(q);

                // 0.5*dt^2*(S_U*dF_dz,Phi_dz).
                loc_rhs_area(i) +=
                  time_step * time_step / 2.0 *
                  (dS_dU[0][0] * dF_dz[0] + dS_dU[0][1] * dF_dz[1]) *
                  fe_values->shape_value(i, q) * fe_values->JxW(q);
                loc_rhs_flux(i) +=
                  time_step * time_step / 2.0 *
                  (dS_dU[1][0] * dF_dz[0] + dS_dU[1][1] * dF_dz[1]) *
                  fe_values->shape_value(i, q) * fe_values->JxW(q);

                // -0.5*dt^2*(H*dF_dz,dPhi_dz).
                loc_rhs_area(i) -= time_step * time_step / 2.0 *
                                   (H[0][0] * dF_dz[0] + H[0][1] * dF_dz[1]) *
                                   dPhi_i_q * fe_values->JxW(q);
                loc_rhs_flux(i) -= time_step * time_step / 2.0 *
                                   (H[1][0] * dF_dz[0] + H[1][1] * dF_dz[1]) *
                                   dPhi_i_q * fe_values->JxW(q);

                // -dt*(S-0.5*dt*S_U*S,Phi_dz).
                loc_rhs_area(i) -=
                  time_step *
                  (S[0] - time_step / 2.0 *
                            (dS_dU[0][0] * S[0] + dS_dU[0][1] * S[1])) *
                  fe_values->shape_value(i, q) * fe_values->JxW(q);
                loc_rhs_flux(i) -=
                  time_step *
                  (S[1] - time_step / 2.0 *
                            (dS_dU[1][0] * S[0] + dS_dU[1][1] * S[1])) *
                  fe_values->shape_value(i, q) * fe_values->JxW(q);
              }
          }
        interval->get_dof_indices(local_dof_indices);

        system_rhs_area.add(local_dof_indices, loc_rhs_area);
        system_rhs_flux.add(local_dof_indices, loc_rhs_flux);
      }

    system_matrix_area.vmult_add(system_rhs_area, old_solution_area);
    system_matrix_flux.vmult_add(system_rhs_flux, old_solution_flux);

    impose_boundary_conditions();
  }

  void
  Vessel1D::impose_boundary_conditions()
  {
    std::map<types::global_dof_index, double> boundary_values_area;
    std::map<types::global_dof_index, double> boundary_values_flux;

    const unsigned int dofs_per_face = 1;

    /// Useful values and retrieve gradients.
    double              A0_inflow  = 0;
    double              A0_outflow = 0;
    std::vector<double> loc_A0(dofs_per_face);

    double              beta_inflow  = 0;
    double              beta_outflow = 0;
    std::vector<double> loc_beta(dofs_per_face);

    double              A_inflow  = 0;
    double              A_outflow = 0;
    std::vector<double> loc_solution_area(dofs_per_face);

    double              Q_inflow  = 0;
    double              Q_outflow = 0;
    std::vector<double> loc_solution_flux(dofs_per_face);

    double dA0_dz_inflow  = 0;
    double dA0_dz_outflow = 0;
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_A0(dofs_per_face);

    double dbeta_dz_inflow  = 0;
    double dbeta_dz_outflow = 0;
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_beta(dofs_per_face);

    double dA_dz_inflow  = 0;
    double dA_dz_outflow = 0;
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_area(dofs_per_face);

    double dQ_dz_inflow  = 0;
    double dQ_dz_outflow = 0;
    std::vector<Tensor<utils::FunctionOnChart::chart_dim, dim, double>>
      loc_grad_flux(dofs_per_face);

    for (auto interval : filter_iterators(dof_handler.active_cell_iterators(),
                                          IteratorFilters::AtBoundary()))
      for (unsigned int face_id = 0; face_id < interval->n_faces(); ++face_id)
        if (interval->face(face_id)->at_boundary())
          {
            const types::boundary_id face_bd_id =
              interval->face(face_id)->boundary_id();

            fe_face_values->reinit(interval, face_id);

            fe_face_values->get_function_values(A0_nodal, loc_A0);
            fe_face_values->get_function_values(beta_nodal, loc_beta);
            fe_face_values->get_function_values(old_solution_area,
                                                loc_solution_area);
            fe_face_values->get_function_values(old_solution_flux,
                                                loc_solution_flux);

            fe_face_values->get_function_gradients(A0_nodal, loc_grad_A0);
            fe_face_values->get_function_gradients(beta_nodal, loc_grad_beta);
            fe_face_values->get_function_gradients(old_solution_area,
                                                   loc_grad_area);
            fe_face_values->get_function_gradients(old_solution_flux,
                                                   loc_grad_flux);

            if (face_bd_id == 0)
              {
                A0_inflow   = loc_A0.front();
                beta_inflow = loc_beta.front();
                A_inflow    = loc_solution_area.front();
                Q_inflow    = loc_solution_flux.front();

                dA0_dz_inflow = domain.project_gradient(loc_grad_A0.front());
                dbeta_dz_inflow =
                  domain.project_gradient(loc_grad_beta.front());
                dA_dz_inflow = domain.project_gradient(loc_grad_area.front());
                dQ_dz_inflow = domain.project_gradient(loc_grad_flux.front());
              }
            else if (face_bd_id == 1)
              {
                A0_outflow   = loc_A0.back();
                beta_outflow = loc_beta.back();
                A_outflow    = loc_solution_area.back();
                Q_outflow    = loc_solution_flux.back();

                dA0_dz_outflow = domain.project_gradient(loc_grad_A0.back());
                dbeta_dz_outflow =
                  domain.project_gradient(loc_grad_beta.back());
                dA_dz_outflow = domain.project_gradient(loc_grad_area.back());
                dQ_dz_outflow = domain.project_gradient(loc_grad_flux.back());
              }
          }

    // psi = beta / A0 * (A^(1/2) - A0^(1/2)).
    const double d_psi_dA0_inflow =
      -beta_inflow *
      ((std::sqrt(A_inflow) - std::sqrt(A0_inflow)) / std::pow(A0_inflow, 2.0) +
       1.0 / (2.0 * std::pow(A0_inflow, 1.5)));
    const double d_psi_dA_inflow =
      beta_inflow / (2.0 * std::sqrt(A_inflow) * A0_inflow);
    const double d_psi_dbeta_inflow =
      (std::sqrt(A_inflow) - std::sqrt(A0_inflow)) / A0_inflow;

    const double d_psi_dA0_outflow =
      -beta_outflow * ((std::sqrt(A_outflow) - std::sqrt(A0_outflow)) /
                         std::pow(A0_outflow, 2.0) +
                       1.0 / (2.0 * std::pow(A0_outflow, 1.5)));
    const double d_psi_dA_outflow =
      beta_outflow / (2.0 * std::sqrt(A_outflow) * A0_outflow);
    const double d_psi_dbeta_outflow =
      (std::sqrt(A_outflow) - std::sqrt(A0_outflow)) / A0_outflow;

    // C1 = int(c1^2, 0, A).
    const double dC1_dA0_outflow = -std::pow(A_outflow, 1.5) * beta_outflow /
                                   (3.0 * std::pow(A0_outflow, 2.0) * density);
    const double dC1_dbeta_outflow =
      std::pow(A_outflow, 1.5) / (3.0 * A0_outflow * density);

    // Matrices evaluated at the boundary_values.
    const std::vector<double> U_inflow{{A_inflow, Q_inflow}};

    const std::vector<double> U_outflow{{A_outflow, Q_outflow}};

    const std::vector<std::vector<double>> H_inflow(
      {{0.0, 1.0},
       {(A_inflow / density) * d_psi_dA_inflow -
          alpha * std::pow((Q_inflow / A_inflow), 2.0),
        2.0 * alpha * Q_inflow / A_inflow}});

    const std::vector<std::vector<double>> H_outflow(
      {{0.0, 1.0},
       {(A_outflow / density) * d_psi_dA_outflow -
          alpha * std::pow((Q_outflow / A_outflow), 2.0),
        2.0 * alpha * Q_outflow / A_outflow}});

    const std::vector<double> dU_dz_inflow({dA_dz_inflow, dQ_dz_inflow});

    const std::vector<double> dU_dz_outflow({dA_dz_outflow, dQ_dz_outflow});

    const std::vector<double> B_inflow(
      {0.0,
       k_r * Q_inflow / A_inflow +
         (A_inflow / density) * (d_psi_dA0_inflow * dA0_dz_inflow +
                                 d_psi_dbeta_inflow * dbeta_dz_inflow)});

    const std::vector<double> B_outflow(
      {0.0,
       k_r * Q_outflow / A_outflow +
         (A_outflow / density) * (d_psi_dA0_outflow * dA0_dz_outflow +
                                  d_psi_dbeta_outflow * dbeta_dz_outflow)});

    // CC = U - dt(H*dU_dz - B).
    const std::vector<double> CC_inflow(
      {U_inflow[0] - time_step * ((H_inflow[0][0] * dU_dz_inflow[0] +
                                   H_inflow[0][1] * dU_dz_inflow[1]) +
                                  B_inflow[0]),
       U_inflow[1] - time_step * ((H_inflow[1][0] * dU_dz_inflow[0] +
                                   H_inflow[1][1] * dU_dz_inflow[1]) +
                                  B_inflow[1])});

    const std::vector<double> CC_outflow{
      {U_outflow[0] - time_step * ((H_outflow[0][0] * dU_dz_outflow[0] +
                                    H_outflow[0][1] * dU_dz_outflow[1]) +
                                   B_outflow[0]),
       U_outflow[1] - time_step * ((H_outflow[1][0] * dU_dz_outflow[0] +
                                    H_outflow[1][1] * dU_dz_outflow[1]) +
                                   B_outflow[1])}};

    const double c_alpha_inflow = std::sqrt(
      beta_inflow / (2.0 * density * A0_inflow) * std::sqrt(A_inflow) +
      std::pow((Q_inflow / A_inflow), 2.0) * alpha * (alpha - 1.0));
    const double c_alpha_outflow = std::sqrt(
      beta_outflow / (2.0 * density * A0_outflow) * std::sqrt(A_outflow) +
      std::pow((Q_outflow / A_outflow), 2.0) * alpha * (alpha - 1.0));

    // Left eigenvectors of matrix H.
    const std::vector<double> L1_outflow(
      {c_alpha_outflow - alpha * Q_outflow / A_outflow, 1.0});
    const std::vector<double> L2_inflow(
      {-c_alpha_inflow - alpha * Q_inflow / A_inflow, 1.0});
    const std::vector<double> L2_outflow(
      {-c_alpha_outflow - alpha * Q_outflow / A_outflow, 1.0});

    // Retrieve the value of the area and the flux to be
    // imposed at the boundary.
    double A_inflow_next  = 0.0;
    double A_outflow_next = 0.0;
    double Q_inflow_next  = 0.0;
    double Q_outflow_next = 0.0;

    // Inflow border conditions.
    // Inflow condition for the source vessel.
    if (bc_inflow_type == BoundaryCondition::Inflow)
      {
        if (source_inflow_type == "Area")
          {
            A_inflow_next = inflow_source_value;

            // If the area is imposed, use Compatibility
            // Conditions on the flux.
            Q_inflow_next =
              (L2_inflow[0] * CC_inflow[0] + L2_inflow[1] * CC_inflow[1]) /
                L2_inflow[1] -
              L2_inflow[0] / L2_inflow[1] * A_inflow_next;

            VectorTools::interpolate_boundary_values(
              dof_handler,
              0,
              Functions::ConstantFunction<dim, double>(Q_inflow_next),
              boundary_values_flux);

            VectorTools::interpolate_boundary_values(
              dof_handler,
              0,
              Functions::ConstantFunction<dim, double>(A_inflow_next),
              boundary_values_area);
          }
        else if (source_inflow_type == "Flux")
          {
            Q_inflow_next = inflow_source_value;

            // If the flux is imposed, use Compatibility
            // Conditions on the area.
            A_inflow_next =
              (L2_inflow[0] * CC_inflow[0] + L2_inflow[1] * CC_inflow[1]) /
                L2_inflow[0] -
              L2_inflow[1] / L2_inflow[0] * Q_inflow_next;

            VectorTools::interpolate_boundary_values(
              dof_handler,
              0,
              Functions::ConstantFunction<dim, double>(A_inflow_next),
              boundary_values_area);

            VectorTools::interpolate_boundary_values(
              dof_handler,
              0,
              Functions::ConstantFunction<dim, double>(Q_inflow_next),
              boundary_values_flux);
          }
      }
    // Inflow condition not for the source vessel, i.e. if in the network.
    // Values given by the Newton system at the nodes.
    else if (bc_inflow_type == BoundaryCondition::BranchingNode)
      {
        VectorTools::interpolate_boundary_values(
          dof_handler,
          0,
          Functions::ConstantFunction<dim, double>(boundary_inflow_area),
          boundary_values_area);

        VectorTools::interpolate_boundary_values(
          dof_handler,
          0,
          Functions::ConstantFunction<dim, double>(boundary_inflow_flux),
          boundary_values_flux);
      }

    // Outflow border conditions.
    // Outflow condition for the vessels in the network. Impose the conditions
    // given by the Newton system at the nodes.
    if (bc_outflow_type == BoundaryCondition::BranchingNode)
      {
        VectorTools::interpolate_boundary_values(
          dof_handler,
          1,
          Functions::ConstantFunction<dim, double>(boundary_outflow_area),
          boundary_values_area);

        VectorTools::interpolate_boundary_values(
          dof_handler,
          1,
          Functions::ConstantFunction<dim, double>(boundary_outflow_flux),
          boundary_values_flux);
      }
    // Outflow condition for the vessels not in the network, i.e. the sinks.
    // Iimpose the non-reflecting conditions.
    else if (bc_outflow_type == BoundaryCondition::NonReflecting)
      {
        std::vector<double> S_outflow(
          {B_outflow[0],
           B_outflow[1] - (dC1_dA0_outflow * dA0_dz_outflow +
                           dC1_dbeta_outflow * dbeta_dz_outflow)});

        A_outflow_next =
          ((CC_outflow[0] + L1_outflow[1] * CC_outflow[1] / L1_outflow[0]) -
           L1_outflow[1] / L1_outflow[0] *
             (L2_outflow[0] / L2_outflow[1] *
                (A_outflow - time_step * S_outflow[0]) +
              (Q_outflow - time_step * S_outflow[1]))) /
          (1 - L1_outflow[1] * L2_outflow[0] / (L1_outflow[0] * L2_outflow[1]));

        Q_outflow_next =
          L2_outflow[0] / L2_outflow[1] *
            (A_outflow - time_step * S_outflow[0] - A_outflow_next) +
          (Q_outflow - time_step * S_outflow[1]);

        VectorTools::interpolate_boundary_values(
          dof_handler,
          1,
          Functions::ConstantFunction<dim, double>(A_outflow_next),
          boundary_values_area);

        VectorTools::interpolate_boundary_values(
          dof_handler,
          1,
          Functions::ConstantFunction<dim, double>(Q_outflow_next),
          boundary_values_flux);
      }

    // Finally apply boundary values.
    MatrixTools::apply_boundary_values(boundary_values_area,
                                       system_matrix_area,
                                       solution_area,
                                       system_rhs_area);

    MatrixTools::apply_boundary_values(boundary_values_flux,
                                       system_matrix_flux,
                                       solution_flux,
                                       system_rhs_flux);
  }

  void
  Vessel1D::solve()
  {
    // Solve for the area.
    solver_vessel_1D.solve(system_matrix_area,
                           solution_area,
                           system_rhs_area,
                           PreconditionIdentity());

    // Solve for the flux.
    solver_vessel_1D.solve(system_matrix_flux,
                           solution_flux,
                           system_rhs_flux,
                           PreconditionIdentity());
  }

  void
  Vessel1D::output_results() const
  {
    if (enable_output && (time_step_number % save_every_n_time_steps) == 0)
      {
        DataOut<utils::FunctionOnChart::chart_dim,
                DoFHandler<utils::FunctionOnChart::chart_dim, dim>>
          data_out;

        DataOutBase::VtkFlags vtk_flags;
        vtk_flags.compression_level =
          DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;

        data_out.set_flags(vtk_flags);

        data_out.attach_dof_handler(dof_handler);

        data_out.add_data_vector(solution_area, "Area");
        data_out.add_data_vector(solution_flux, "Flux");

        data_out.build_patches();

        const std::string filename =
          Core::prm_output_directory + "/solution" +
          Utilities::int_to_string(vessel_id, 3) + "-" +
          Utilities::int_to_string(time_step_number / save_every_n_time_steps,
                                   6) +
          ".vtu";

        std::ofstream output(filename);

        data_out.write_vtu(output);

        data_out.clear();
      }
  }

  bool
  Vessel1D::check_CFL() const
  {
    bool CFL_is_satisfied = true;

    // Save local quantities.
    const unsigned int dofs_per_face = 1;

    // Useful values.
    std::vector<double> loc_A0(dofs_per_face);
    std::vector<double> loc_beta(dofs_per_face);
    std::vector<double> loc_solution_area_next(dofs_per_face);
    std::vector<double> loc_solution_flux_next(dofs_per_face);
    std::vector<double> loc_solution_area(dofs_per_face);
    std::vector<double> loc_solution_flux(dofs_per_face);

    // Loop over intervals.
    for (auto interval : dof_handler.active_cell_iterators())
      {
        const double h = interval->diameter();

        fe_face_values->reinit(interval, 0);

        fe_face_values->get_function_values(A0_nodal, loc_A0);
        fe_face_values->get_function_values(beta_nodal, loc_beta);
        fe_face_values->get_function_values(solution_area,
                                            loc_solution_area_next);
        fe_face_values->get_function_values(solution_flux,
                                            loc_solution_flux_next);
        fe_face_values->get_function_values(old_solution_area,
                                            loc_solution_area);
        fe_face_values->get_function_values(old_solution_flux,
                                            loc_solution_flux);

        const double A0     = loc_A0.front();
        const double beta   = loc_beta.front();
        const double A_next = loc_solution_area_next.front();
        const double Q_next = loc_solution_flux_next.front();
        const double A      = loc_solution_area.front();
        const double Q      = loc_solution_flux.front();

        const double d_psi_dA_next = beta / (2.0 * std::sqrt(A_next) * A0);
        const double lambda_1_next =
          alpha * (Q_next / A_next) +
          std::sqrt((A_next / density) * d_psi_dA_next +
                    alpha * (alpha - 1.0) * std::pow(Q_next / A_next, 2.0));

        const double d_psi_dA = beta / (2.0 * std::sqrt(A) * A0);
        const double lambda_1 =
          alpha * (Q / A) +
          std::sqrt((A / density) * d_psi_dA +
                    alpha * (alpha - 1.0) * std::pow(Q / A, 2.0));

        const double max_lambda =
          (lambda_1_next < lambda_1) ? lambda_1 : lambda_1_next;

        CFL_is_satisfied =
          (time_step < std::sqrt(3.0) / 3.0 * h / max_lambda) &&
          CFL_is_satisfied;

        if (!CFL_is_satisfied)
          {
            pcout << std::to_string(time_step) + " > " +
                       std::to_string(std::sqrt(3.0) / 3 * h / max_lambda)
                  << std::endl;

            return false;
          }
      }

    return CFL_is_satisfied;
  }

} // namespace lifex