## ---------------------------------------------------------------------
## Copyright (C) 2019 - 2023 by the lifex authors.
##
## This file is part of lifex.
##
## lifex is free software; you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## lifex is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with lifex.  If not, see <http://www.gnu.org/licenses/>.
## ---------------------------------------------------------------------

lifex_add_executable(EXAMPLE vessels_network_1D vessels_network_1D_run.cpp vessels_network_1D.cpp vessel_1D.cpp mesh_handler_1D.cpp function_on_chart.cpp)
lifex_link_libraries(EXAMPLE vessels_network_1D)