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

#ifndef LIFEX_EXAMPLES_VESSELS_NETWORK_1D_HPP_
#define LIFEX_EXAMPLES_VESSELS_NETWORK_1D_HPP_

#include "core/source/core_model.hpp"
#include "core/source/init.hpp"

#include "examples/vessels_network_1D/function_on_chart.hpp"
#include "examples/vessels_network_1D/vessel_1D.hpp"

#include <deal.II/base/parsed_function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/matrix_tools.h>

#include <dirent.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lifex::examples
{
  /**
   * @brief 1D hemodynamics in a vessels' network.
   *
   * Class that manages the hemodynamics in a network of vessels by
   * projecting N-S equations in 1D. The geometry of the problem is embedded in
   * the physical 3D space. Balance equations couple the vessels ingoing and
   * outgoing the same node of the network.
   *
   * The network is read in input from a vtk file containing its topology,
   * namely the connectivity map and the physical coordinates of the vessels'
   * extreme points. Branching is not arbitrary since at every node only one
   * ingoing vessel and two outgoing ones are allowed in this model.
   *
   * Boundary conditions can be arbitrarily prescribed by the user.
   *
   * References:
   * - https://www.mate.polimi.it/biblioteca/add/qmox/mox08.pdf
   */
  class VesselsNetwork : public CoreModel
  {
  public:
    /// Constructor.
    VesselsNetwork(const std::string &subsection);

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// Override of @ref CoreModel::run.
    virtual void
    run() override;

  protected:
    /// Initialize every vessel of the network.
    void
    parse_vessels_network(ParamHandler &params);

    /// Read the domain from a vtk file and initialize the connectivity map and
    /// the extremal points coordinates and ID of the vessels.
    void
    read_domain();

    /// @brief Check the consistency of the network with respect to the model
    /// solved.
    ///
    /// Namely, these conditions are checked:
    /// - one only source vessel is allowed;
    /// - at every branching node, only a 2-way branch is allowed.
    void
    set_connectivity();

    /// Initialize each vessel in the network calling the
    /// @ref Vessel1D::setup_system method.
    void
    setup_vessels_network();

    /// Evaluate the inflow condition given in input and set the value for the
    /// source vessel at its inflow node.
    void
    impose_inflow_condition();

    /// Solve the equations with Newton's method at the branching nodes and
    /// impose the values computed.
    void
    branching_conditions();

    /// Function limited for internal use and debugging. It prints to the
    /// terminal the values of this class' data member.
    void
    print_network_state() const;

    /// Physical time of the simulation [s].
    double time;
    /// Number of dicrete time steps elapsed.
    unsigned int timestep_number;

    /// @name Parameters read from file.
    /// @{

    /// Path to the directory containing vessels' parameters files.
    std::string prm_path_to_vessels_files_directory;

    /// Initial time of the simulation [s].
    double prm_time_init;
    /// Final time of the simulation [s].
    double prm_time_final;
    /// Length of the time step used in the time-homogeneous numerical scheme
    /// [s].
    double prm_time_step;

    /// Finite elements polynomial degree.
    unsigned int prm_fe_degree;

    /// Alpha [1], where @f$ \alpha = \frac{\int_\mathcal{S}s^2\,d\sigma}{A}
    /// @f$, and @f$ s @f$ is the profile law chosen.
    double prm_alpha;
    /// Density [g cm^-3] of the blood over the whole network. It overwrites the
    /// density if specified in each single vessel.
    double prm_density;
    /// Friction [cm^2 s^-1]. It depends on viscosity.
    double prm_K_r;

    /// Parsed functions representing the inflow Dirichlet condition on the
    /// area.
    Functions::ParsedFunction<dim> prm_inflow_condition;
    /// Choose what to impose the inflow. Available options are "Area", "Flux".
    std::string prm_inflow_type;

    /// Value of parameter @f$\gamma_1@f$ [1] for the vessel approaching the
    /// node. Default is zero, namely no dissipation at the node.
    double prm_gamma_1;
    /// Value of parameter @f$\gamma_2@f$ [1] for the first vessel leaving the
    /// node. Default is zero, namely no dissipation at the node.
    double prm_gamma_2;
    /// Value of parameter @f$\gamma_2@f$ [1] for the second vessel leaving
    /// the node. Default is zero, namely no dissipation at the node.
    double prm_gamma_3;

    /// Relative tolerance used in the newton solver.
    double prm_newton_tolerance;
    /// Maximum number of iterations used in the newton solver.
    unsigned int prm_newton_max_iterations;

    /// Name of the vtk file containing the network mesh.
    std::string prm_network_mesh_filename;

    /// Choose wheter to save the output files.
    bool prm_enable_output;
    /// Save every n time_steps the output.
    unsigned int prm_save_every_n_time_steps;

    /// @}

    /// Main data structure containing vessels. They are indexed sequentially
    /// starting from 0, which is also were the inflow condition is imposed.
    /// @note We store pointers because we would need the copy constructor
    /// otherwise, which might be ill-formed.
    std::vector<std::unique_ptr<lifex::Vessel1D>> vessels;
    /// Data structure that links nodes' IDs to their physical coordinates.
    /// Position 0 is reserved for the source node.
    std::vector<Point<dim>> network_id_coordinates;
    /// Data structure that links the node ID with the vessels' ones
    /// intersecting in that node.
    std::vector<std::vector<unsigned int>> connectivity_map;

    /// Number of vessels.
    unsigned int n_vessels;
    /// Number of nodes in the mesh.
    unsigned int n_nodes;
  };

} // namespace lifex::examples

#endif // LIFEX_EXAMPLES_VESSELS_NETWORK_1D_HPP_
