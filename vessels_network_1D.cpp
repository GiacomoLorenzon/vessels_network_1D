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

#include "examples/vessels_network_1D/vessels_network_1D.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <regex>
#include <utility>

namespace lifex::examples
{
  VesselsNetwork::VesselsNetwork(const std::string &subsection)
    : CoreModel(subsection)
    , time(0.0)
    , timestep_number(0u)
    , n_vessels(0u)
    , n_nodes(0u)
  {}

  void
  VesselsNetwork::declare_parameters(ParamHandler &params) const
  {
    // Directory containing the vessel files.
    params.declare_entry("Directory of the vessels parameters",
                         "./",
                         Patterns::Anything(),
                         "Path to the directory in which the vessel parameter "
                         "files are saved.");

    // Temporal parameters subsection.
    params.enter_subsection("Temporal parameters");
    {
      params.declare_entry("Initial time",
                           "0.0",
                           Patterns::Double(0.0),
                           "Value of initial time [s].");

      params.declare_entry(
        "Final time",
        "0.01",
        Patterns::Double(0.0),
        "Value of time at which the simulation is stopped [s].");

      params.declare_entry("Time step",
                           "1e-4",
                           Patterns::Double(0.0),
                           "Length of the discrete time step [s].");
    }
    params.leave_subsection();

    params.declare_entry("Finite element polynomial degree",
                         "1",
                         Patterns::Integer(1, 2),
                         "Polynomial degree of the finite element space.");

    // Model parameters common to all vessels in the network.
    params.enter_subsection("Blood flow model parameters");
    {
      params.declare_entry(
        "Alpha",
        "1.0",
        Patterns::Double(1.0),
        "Momentum flux correction coefficient (Coriolis coefficient) [1].");
      params.declare_entry("Density",
                           "1.05",
                           Patterns::Double(0.0),
                           "Value of the density of the blood [g cm^-3].");
      params.declare_entry(
        "Friction",
        "2.4190",
        Patterns::Double(0.0),
        "Friction parameter due to blood viscosity [cm^2 s^-1]. It is "
        "a function of the blood profile law chosen. See "
        "references for further explanations.");
    }
    params.leave_subsection();

    // Impose boundary condition area subsection.
    params.enter_subsection("Boundary condition");
    {
      params.declare_entry_selection("Inflow condition type",
                                     "Area",
                                     "Area|Flux");

      // Inflow condition subsection.
      params.enter_subsection("Inflow condition expression");
      {
        // Mathematical expression of the inflow flux or area. The function
        // should depend on the physical 3D space coordinates and on time.
        Functions::ParsedFunction<dim>::declare_parameters(params);
      }
      params.leave_subsection();
    }
    params.leave_subsection();

    // Parameters for branching.
    params.enter_subsection("Branching parameters");
    {
      params.declare_entry(
        "Gamma 1",
        "0.0",
        Patterns::Double(0.0),
        "Value of parameter Gamma for the vessel approaching the node.");

      params.declare_entry(
        "Gamma 2",
        "0.0",
        Patterns::Double(0.0),
        "Value of parameter Gamma for the first vessel leaving the node.");

      params.declare_entry(
        "Gamma 3",
        "0.0",
        Patterns::Double(0.0),
        "Value of parameter Gamma for the second vessel leaving the node.");
    }
    params.leave_subsection();

    // Parameters for newton solver.
    params.enter_subsection("Newton solver for branching conditions");
    {
      params.declare_entry(
        "Relative tolerance",
        "1e-5",
        Patterns::Double(0.0),
        "Value of the relative tolerance used in the Newton solver.");

      params.declare_entry(
        "Iterations",
        "1000",
        Patterns::Integer(0),
        "Maximum number of iterations used in the Newton solver.");
    }
    params.leave_subsection();

    params.declare_entry("Mesh file",
                         "",
                         Patterns::Anything(),
                         "Name of the VTK file containing the mesh.");

    params.enter_subsection("Output");
    {
      params.declare_entry("Enable output",
                           "true",
                           Patterns::Bool(),
                           "Decide whether save the output files.");

      params.declare_entry("Print VTU every n time_steps",
                           "1",
                           Patterns::Integer(1),
                           "VTU is written only once every n time_steps.");
    }
    params.leave_subsection();
  }

  void
  VesselsNetwork::parse_parameters(ParamHandler &params)
  {
    params.parse();

    // Read the name of the directory containing the vessel files.
    prm_path_to_vessels_files_directory =
      params.get("Directory of the vessels parameters") + "/";

    // Read temporal parameters.
    params.enter_subsection("Temporal parameters");
    {
      prm_time_init  = params.get_double("Initial time");
      prm_time_final = params.get_double("Final time");
      prm_time_step  = params.get_double("Time step");
    }
    params.leave_subsection();

    // Read finite element degree parameter.
    prm_fe_degree = params.get_integer("Finite element polynomial degree");


    // Model parameters common to all vessels in the network.
    params.enter_subsection("Blood flow model parameters");
    {
      prm_alpha   = params.get_double("Alpha");
      prm_density = params.get_double("Density");
      prm_K_r     = params.get_double("Friction");
    }
    params.leave_subsection();

    // Read boundary condition subsection.
    params.enter_subsection("Boundary condition");
    {
      prm_inflow_type = params.get("Inflow condition type");

      // Read inflow condition area.
      params.enter_subsection("Inflow condition expression");
      {
        prm_inflow_condition.parse_parameters(params);
      }
      params.leave_subsection();
    }
    params.leave_subsection();

    // Parameters for branching.
    params.enter_subsection("Branching parameters");
    {
      prm_gamma_1 = params.get_double("Gamma 1");
      prm_gamma_2 = params.get_double("Gamma 2");
      prm_gamma_3 = params.get_double("Gamma 3");
    }
    params.leave_subsection();

    // Parameters for newton solver.
    params.enter_subsection("Newton solver for branching conditions");
    {
      prm_newton_tolerance      = params.get_double("Relative tolerance");
      prm_newton_max_iterations = params.get_integer("Iterations");
    }
    params.leave_subsection();

    // Read the topology of the network.
    prm_network_mesh_filename = params.get("Mesh file");

    params.enter_subsection("Output");
    {
      prm_enable_output = params.get_bool("Enable output");
      prm_save_every_n_time_steps =
        params.get_integer("Print VTU every n time_steps");
    }
    params.leave_subsection();

    // Parsing for each vessel in the network.
    parse_vessels_network(params);
  }

  void
  VesselsNetwork::run()
  {
    // Read the topology from the file given in input.
    read_domain();

    // Initialize each vessel in the network.
    setup_vessels_network();

    // Iterate.
    while (time < prm_time_final)
      {
        time += prm_time_step;
        ++timestep_number;

        pcout << "\nTimestep " << timestep_number << ", time = " << time
              << std::endl;

        // Impose the boundary condition at the source vessel.
        impose_inflow_condition();

        // Solve node equations and impose boundary data.
        branching_conditions();

        // Loop over the vessels.
        for (auto &ptr_vessel : vessels)
          ptr_vessel->solve_time_step();
      }
  }

  void
  VesselsNetwork::parse_vessels_network(ParamHandler &params)
  {
    // Automatically counts the number of vessels parameters file.
    std::filesystem::path directory_name(prm_path_to_vessels_files_directory);

    // Check that the directory exists.
    if (!std::filesystem::is_directory(directory_name))
      throw std::runtime_error("Directory not found. Please create it or "
                               "check if the path is correct: " +
                               prm_path_to_vessels_files_directory);

    // Count files named "vesselXXX.prm", where XXX stands for a number.
    const std::regex pattern("vessel\\d+\\.prm");
    n_vessels =
      std::count_if(std::filesystem::directory_iterator(directory_name),
                    std::filesystem::directory_iterator{},
                    [&](const std::filesystem::directory_entry &entry) {
                      const std::string filename =
                        entry.path().filename().string();
                      return std::filesystem::is_regular_file(entry) &&
                             std::regex_match(filename, pattern);
                    });


    Assert(n_vessels, ExcMessage("No vesselsXXX.prm files found."));

    vessels.reserve(n_vessels);

    // Initialize each vessel with increasing ID, starting from 0.
    for (unsigned int vessel_ID = 0u; vessel_ID < n_vessels; vessel_ID++)
      {
        vessels.emplace_back(std::make_unique<Vessel1D>(
          prm_path_to_vessels_files_directory + "vessel" +
            Utilities::int_to_string(vessel_ID, 3) + ".prm",
          vessel_ID,
          prm_time_step,
          prm_time_init,
          prm_time_final,
          prm_fe_degree,
          prm_alpha,
          prm_density,
          prm_K_r,
          prm_inflow_type,
          prm_enable_output,
          prm_save_every_n_time_steps));

        vessels.at(vessel_ID)->declare_parameters(params);
        vessels.at(vessel_ID)->parse_parameters(params);
      }
  }

  void
  VesselsNetwork::read_domain()
  {
    // Open the .vtk file containing the network topology, read it and save in a
    // triangulation object.
    std::ifstream in(prm_network_mesh_filename);

    if (in)
      {
        Triangulation<utils::FunctionOnChart::chart_dim, dim> network;
        GridIn<utils::FunctionOnChart::chart_dim, dim>        grid_in(network);
        grid_in.read_vtk(in);

        n_nodes = network.n_vertices();
        // global index nodes are sequential.
        network_id_coordinates.resize(n_nodes);
        connectivity_map.resize(n_nodes);

        unsigned int vessel_ID = 0;

        for (const auto &edge : network.active_cell_iterators())
          {
            // Get the necessary info to build the local triangulations.

            // Retrieve the global vertex numbering.
            int global_index_inflow  = edge->vertex_index(0);
            int global_index_outflow = edge->vertex_index(1);

            network_id_coordinates.at(global_index_inflow) =
              network.get_vertices()[global_index_inflow];
            network_id_coordinates.at(global_index_outflow) =
              network.get_vertices()[global_index_outflow];

            // Update the connectivity map and set the extremal point for
            // the considered vessel.
            connectivity_map.at(global_index_inflow).emplace_back(vessel_ID);
            connectivity_map.at(global_index_outflow).emplace_back(vessel_ID);

            vessels.at(vessel_ID)->set_extremal_points(
              std::make_pair(global_index_inflow,
                             network_id_coordinates.at(global_index_inflow)),
              std::make_pair(global_index_outflow,
                             network_id_coordinates.at(global_index_outflow)));

            vessel_ID++;
          }
      }
    else // if (!in)
      {
        Assert(false, ExcFileNotOpen(prm_network_mesh_filename));
      }

    set_connectivity();
  }

  void
  VesselsNetwork::set_connectivity()
  {
    bool         source_found = false;
    unsigned int this_node_ID = 0;

    for (const auto &vessels_IDs_at_this_node : connectivity_map)
      {
        // In this model we manage just the splitting of a single
        // vessel in up to two other vessels.
        Assert(vessels_IDs_at_this_node.size() == 3 ||
                 vessels_IDs_at_this_node.size() == 1,
               ExcMessage(
                 "This model accounts only for 1 ingoing and 2 "
                 "outgoing vessels at each network node. Please check the "
                 "topology of your network. Node ID: " +
                 std::to_string(this_node_ID) + " intersects " +
                 std::to_string(vessels_IDs_at_this_node.size()) +
                 " vessel(s)."));

        // If the current node is a branching node.
        if (vessels_IDs_at_this_node.size() == 3)
          {
            // Get which vessel is ingoing into the node.
            bool is_vessel_1_ingoing =
              (vessels.at(vessels_IDs_at_this_node[0])
                 ->get_id_extremal_point(Vessel1D::ExtremalPoint::Outflow) ==
               this_node_ID);
            bool is_vessel_2_ingoing =
              (vessels.at(vessels_IDs_at_this_node[1])
                 ->get_id_extremal_point(Vessel1D::ExtremalPoint::Outflow) ==
               this_node_ID);
            bool is_vessel_3_ingoing =
              (vessels.at(vessels_IDs_at_this_node[2])
                 ->get_id_extremal_point(Vessel1D::ExtremalPoint::Outflow) ==
               this_node_ID);

            // Check that exactly one vessel is entering the node. Logic
            // condition follows. Explanation:
            // - xor operator is firstly exploited. (is_vessel_1_ingoing ^
            // is_vessel_2_ingoing ^ is_vessel_3_ingoing) is thus true if there
            // exists only one variable set to true, or each one of them is
            // true.
            // - One must then exclude that the three vessels are not ingoing
            // simultaneously.
            Assert(is_vessel_1_ingoing ^ is_vessel_2_ingoing ^
                       is_vessel_3_ingoing &&
                     is_vessel_1_ingoing + is_vessel_2_ingoing +
                         is_vessel_3_ingoing !=
                       3,
                   ExcMessage(
                     "This model accounts only for 1 ingoing and 2 "
                     "outgoing vessels at each network node. Please check the "
                     "topology of your network. At node ID: " +
                     std::to_string(this_node_ID) + "\n\tvessel ID: " +
                     std::to_string(vessels_IDs_at_this_node[0]) + "is " +
                     (is_vessel_1_ingoing ? "in" : "out") + "going." +
                     "\n\tvessel ID: " +
                     std::to_string(vessels_IDs_at_this_node[1]) + "is " +
                     (is_vessel_2_ingoing ? "in" : "out") + "going." +
                     "\n\tvessel ID: " +
                     std::to_string(vessels_IDs_at_this_node[2]) + "is " +
                     (is_vessel_3_ingoing ? "in" : "out") + "going."));

            // Impose for later computations what values they must override at
            // the boundary. If is_vessel_i_ingoing == true, outflow conditions
            // will be set after solving balance equations at the node.
            // Otherwise, inflow conditions will be overridden.
            vessels.at(vessels_IDs_at_this_node[0])
              ->trigger_branching_bc(is_vessel_1_ingoing);
            vessels.at(vessels_IDs_at_this_node[1])
              ->trigger_branching_bc(is_vessel_2_ingoing);
            vessels.at(vessels_IDs_at_this_node[2])
              ->trigger_branching_bc(is_vessel_3_ingoing);
          }
        else // if(vessels_IDs_at_this_node.size() == 1)
          {
            if (vessels_IDs_at_this_node[0] == 0)
              source_found = true;
            else
              Assert(vessels.at(vessels_IDs_at_this_node[0])
                         ->get_id_extremal_point(
                           Vessel1D::ExtremalPoint::Inflow) != 0,
                     ExcMessage(
                       "Only one vessel source is allowed for this model. ID "
                       "0 is reserved for the source node."));
          }

        this_node_ID++;
      }

    if (!source_found)
      Assert(false,
             ExcMessage(
               "Vessel source not found. One vessel source is required."));
  }

  void
  VesselsNetwork::setup_vessels_network()
  {
    // Save the topology.
    for (auto &vessel_ptr : vessels)
      {
        // Initialize each vessel.
        vessel_ptr->setup_system();
      }
  }

  void
  VesselsNetwork::impose_inflow_condition()
  {
    prm_inflow_condition.set_time(time);

    vessels.at(0u)->set_inflow_source_value(
      prm_inflow_condition.value(network_id_coordinates.at(0u)));
  }

  void
  VesselsNetwork::branching_conditions()
  {
    unsigned int this_node_ID = 0;
    // For each node of the network.
    for (const auto &vessels_IDs_at_this_node : connectivity_map)
      {
        // Skip iteration for source and sinks.
        if (vessels_IDs_at_this_node.size() != 3)
          continue;

        // Save data for clarity.
        unsigned int              ingoing_vessel_ID = 0;
        std::vector<unsigned int> outgoing_vessel_IDs;

        // Associate and distinguish ingoing and outgoing vessels.
        for (const auto &this_vessel_ID : vessels_IDs_at_this_node)
          {
            if (vessels.at(this_vessel_ID)
                  ->get_id_extremal_point(Vessel1D::ExtremalPoint::Outflow) ==
                this_node_ID)
              ingoing_vessel_ID = this_vessel_ID;
            else
              outgoing_vessel_IDs.emplace_back(this_vessel_ID);
          }

        // Get numerical values from the ingoing vessel.
        bool                   is_ingoing = true;
        Vessel1D::BoundaryData in_1 =
          vessels.at(ingoing_vessel_ID)->get_boundary_data(is_ingoing);

        // Get numerical values from the first outgoing vessel.
        is_ingoing = false;
        Vessel1D::BoundaryData out_2 =
          vessels.at(outgoing_vessel_IDs[0])
            ->get_boundary_data(is_ingoing /* = false */);
        double cos_alpha_out_2 = in_1.direction * out_2.direction;
        Assert(cos_alpha_out_2 >= 0 || (prm_gamma_1 == 0 && prm_gamma_2 == 0),
               ExcMessage("The angle between two vessels can't be greater than "
                          "90 degrees. The error comes from:\n"
                          " - Vessel (ID: " +
                          std::to_string(ingoing_vessel_ID) +
                          ")\n - Vessel (ID: " +
                          std::to_string(outgoing_vessel_IDs[0]) + ")\n"));

        // Get numerical values from the second outgoing vessel.
        Vessel1D::BoundaryData out_3 =
          vessels.at(outgoing_vessel_IDs[1])
            ->get_boundary_data(is_ingoing /* = false */);
        double cos_alpha_out_3 = in_1.direction * out_3.direction;
        Assert(cos_alpha_out_2 >= 0 || (prm_gamma_1 == 0 && prm_gamma_3 == 0),
               ExcMessage("The angle between two vessels can't be greater than "
                          "90 degrees. The error comes from:\n"
                          " - Vessel (ID: " +
                          std::to_string(ingoing_vessel_ID) +
                          ")\n - Vessel (ID: " +
                          std::to_string(outgoing_vessel_IDs[1]) + ")\n"));
        // @TODO: test more extensively for alpha>pi/4

        // All the data are now present, defining matrix and vector for Newton
        // using previous time step data as initial guess.
        Vector<double>     prev_solution({in_1.U[0],
                                      out_2.U[0],
                                      out_3.U[0],
                                      in_1.U[1],
                                      out_2.U[1],
                                      out_3.U[1]});
        Vector<double>     solution({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        Vector<double>     b({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        FullMatrix<double> newton_matrix(6, 6);
        Vector<double>     delta({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        // Store lambdas for easy computations.
        auto compute_CA = [&](const bool &                  is_inflow,
                              const Vessel1D::BoundaryData &data,
                              const double &                gamma,
                              const double &                cos_alpha = 0.0) {
          return (is_inflow) ?

                   data.beta /
                       (2.0 * prm_density * std::sqrt(data.U[0]) * data.A0) -
                     std::pow(data.U[1], 2) / std::pow(data.U[0], 3) *
                       (1.0 - 2.0 * gamma * std::signbit(data.U[1])) :

                   data.beta /
                       (2.0 * prm_density * std::sqrt(data.U[0]) * data.A0) -
                     std::pow(data.U[1], 2) / std::pow(data.U[0], 3) *
                       (1.0 + 2.0 * gamma * std::sqrt(2.0 * (1.0 - cos_alpha)) *
                                std::signbit(data.U[1]));
        };

        auto compute_CQ = [&](const bool &                  is_inflow,
                              const Vessel1D::BoundaryData &data,
                              const double &                gamma,
                              const double &                cos_alpha = 0.0) {
          return (is_inflow) ?

                   data.U[1] / std::pow(data.U[0], 2) *
                     (1.0 - 2.0 * gamma * std::signbit(data.U[1])) :

                   data.U[1] / std::pow(data.U[0], 2) *
                     (1.0 + 2.0 * gamma * std::sqrt(2.0 * (1.0 - cos_alpha)) *
                              std::signbit(data.U[1]));
        };

        auto compute_b = [&](const bool &                  is_inflow,
                             const Vessel1D::BoundaryData &data,
                             const double &                gamma,
                             const double &                cos_alpha = 0.0) {
          return (is_inflow) ?

                   data.beta * (std::sqrt(data.U[0]) - 2 * std::sqrt(data.A0)) /
                       (2 * prm_density * data.A0) +
                     0.5 * std::pow(data.U[1], 2) / std::pow(data.U[0], 2) *
                       (1.0 - 2.0 * gamma * std::signbit(data.U[1])) :

                   data.beta * (std::sqrt(data.U[0]) - 2 * std::sqrt(data.A0)) /
                       (2 * prm_density * data.A0) +
                     0.5 * std::pow(data.U[1], 2) / std::pow(data.U[0], 2) *
                       (1.0 + 2.0 * gamma * std::sqrt(2.0 * (1.0 - cos_alpha)) *
                                std::signbit(data.U[1]));
        };

        unsigned int iter = 0;
        do // Newton iterations.
          {
            iter++;

            const double &b_in_1 = compute_b(true, in_1, prm_gamma_1);
            const double &b_out_2 =
              compute_b(false, out_2, prm_gamma_2, cos_alpha_out_2);
            const double &b_out_3 =
              compute_b(false, out_3, prm_gamma_3, cos_alpha_out_3);

            const double &CA_in_1 = compute_CA(true, in_1, prm_gamma_1);
            const double &CA_out_2 =
              compute_CA(false, out_2, prm_gamma_2, cos_alpha_out_2);
            const double &CA_out_3 =
              compute_CA(false, out_3, prm_gamma_3, cos_alpha_out_3);

            const double &CQ_in_1 = compute_CQ(true, in_1, prm_gamma_1);
            const double &CQ_out_2 =
              compute_CQ(false, out_2, prm_gamma_2, cos_alpha_out_2);
            const double &CQ_out_3 =
              compute_CQ(false, out_3, prm_gamma_3, cos_alpha_out_3);

            // Right-hand side.
            b(0) = 0;
            b(1) = b_out_2 - b_in_1;
            b(2) = b_out_3 - b_in_1;
            b(3) = in_1.L1 * in_1.CC;
            b(4) = out_2.L2 * out_2.CC;
            b(5) = out_3.L2 * out_3.CC;
            // Matrix, row 0.
            newton_matrix(0, 0) = 0;
            newton_matrix(0, 1) = 0;
            newton_matrix(0, 2) = 0;
            newton_matrix(0, 3) = 1;
            newton_matrix(0, 4) = -1;
            newton_matrix(0, 5) = -1;
            // Matrix, row 1.
            newton_matrix(1, 0) = CA_in_1;
            newton_matrix(1, 1) = -CA_out_2;
            newton_matrix(1, 2) = 0;
            newton_matrix(1, 3) = CQ_in_1;
            newton_matrix(1, 4) = -CQ_out_2;
            newton_matrix(1, 5) = 0;
            // Matrix, row 2.
            newton_matrix(2, 0) = CA_in_1;
            newton_matrix(2, 1) = 0;
            newton_matrix(2, 2) = -CA_out_3;
            newton_matrix(2, 3) = CQ_in_1;
            newton_matrix(2, 4) = 0;
            newton_matrix(2, 5) = -CQ_out_3;
            // Matrix, row 3.
            newton_matrix(3, 0) = in_1.L1[0];
            newton_matrix(3, 1) = 0;
            newton_matrix(3, 2) = 0;
            newton_matrix(3, 3) = in_1.L1[1];
            newton_matrix(3, 4) = 0;
            newton_matrix(3, 5) = 0;
            // Matrix, row 4.
            newton_matrix(4, 0) = 0;
            newton_matrix(4, 1) = out_2.L2[0];
            newton_matrix(4, 2) = 0;
            newton_matrix(4, 3) = 0;
            newton_matrix(4, 4) = out_2.L2[1];
            newton_matrix(4, 5) = 0;
            // Matrix, row 5.
            newton_matrix(5, 0) = 0;
            newton_matrix(5, 1) = 0;
            newton_matrix(5, 2) = out_3.L2[0];
            newton_matrix(5, 3) = 0;
            newton_matrix(5, 4) = 0;
            newton_matrix(5, 5) = out_3.L2[1];

            // Invert the matrix. gauss_jordan() is implicitely invoked.
            newton_matrix.invert(newton_matrix);
            newton_matrix.vmult(solution, b, false);

            // Compute the relative error: (solution -
            // prev_solution)/min_norm.
            delta *= 0;
            delta.add(1.0, solution, -1.0, prev_solution);

            // Update the solution.
            prev_solution = solution;
          }
        while (delta.linfty_norm() / std::min(solution.linfty_norm(),
                                              prev_solution.linfty_norm()) >
                 prm_newton_tolerance &&
               iter < prm_newton_max_iterations);

        // Set the values found.
        vessels.at(ingoing_vessel_ID)
          ->set_boundary_values({solution[0], solution[3]},
                                Vessel1D::ExtremalPoint::Outflow);
        vessels.at(outgoing_vessel_IDs[0])
          ->set_boundary_values({solution[1], solution[4]},
                                Vessel1D::ExtremalPoint::Inflow);
        vessels.at(outgoing_vessel_IDs[1])
          ->set_boundary_values({solution[2], solution[5]},
                                Vessel1D::ExtremalPoint::Inflow);

        this_node_ID++;
      }
  }

  void
  VesselsNetwork::print_network_state() const
  {
    pcout << "=============================================" << std::endl;
    pcout << " NETWORK INFO" << std::endl;
    pcout << "=============================================" << std::endl;
    pcout << std::endl;
    pcout << " time           : " << time << std::endl;
    pcout << " timestep_number: " << timestep_number << std::endl;
    pcout << std::endl;

    pcout << " path: " << prm_path_to_vessels_files_directory << std::endl;
    pcout << std::endl;

    pcout << " prm_time_init     : " << prm_time_init << std::endl;
    pcout << " prm_time_final    : " << prm_time_final << std::endl;
    pcout << " prm_time_step     : " << prm_time_step << std::endl;
    pcout << std::endl;

    pcout << " prm_alpha  : " << prm_alpha << std::endl;
    pcout << " prm_density: " << prm_density << std::endl;
    pcout << " prm_K_r    : " << prm_K_r << std::endl;

    pcout << std::endl;

    pcout << " prm_gamma_1: " << prm_gamma_1 << std::endl;
    pcout << " prm_gamma_2: " << prm_gamma_2 << std::endl;
    pcout << " prm_gamma_3: " << prm_gamma_3 << std::endl;
    pcout << std::endl;

    pcout << " prm_newton_tolerance     : " << prm_newton_tolerance
          << std::endl;
    pcout << " prm_newton_max_iterations: " << prm_newton_max_iterations
          << std::endl;
    pcout << std::endl;

    pcout << " prm_network_mesh_filename: " << prm_network_mesh_filename
          << std::endl;
    pcout << std::endl;

    pcout << " prm_enable_output: " << prm_enable_output << std::endl;
    pcout << " prm_save_every_n_time_steps: " << prm_save_every_n_time_steps
          << std::endl;
    pcout << std::endl;

    pcout << " n_vessels : " << n_vessels << std::endl;
    pcout << " n_nodes   : " << n_nodes << std::endl;
    pcout << std::endl;

    pcout << "=============================================" << std::endl;
    pcout << "=============================================" << std::endl;
    pcout << " VESSELS' INFO" << std::endl;
    pcout << "=============================================" << std::endl;

    for (const auto &ptr : vessels)
      {
        ptr->print_vessel_state();
        ptr->print_triangulation_info();
      }

    pcout << "=============================================" << std::endl;
    pcout << " CONNECTIVITY MAP" << std::endl;
    pcout << "=============================================" << std::endl;

    unsigned int this_node_ID = 0;
    for (const auto &vessels_IDs_at_this_node : connectivity_map)
      {
        pcout << "\nID node (*): " << Utilities::int_to_string(this_node_ID, 3);
        for (const auto &id : vessels_IDs_at_this_node)
          {
            bool is_ingoing =
              vessels.at(id)->get_id_extremal_point(
                Vessel1D::ExtremalPoint::Outflow) == this_node_ID;

            pcout << "\n └ ID vessel: " << Utilities::int_to_string(id, 3)
                  << ((is_ingoing) ? " |   ingoing -->(*)" :
                                     " |  outgoing    (*)-->");
          }
        pcout << std::endl;
        this_node_ID++;
      }
    pcout << std::endl;
  }
} // namespace lifex::examples
