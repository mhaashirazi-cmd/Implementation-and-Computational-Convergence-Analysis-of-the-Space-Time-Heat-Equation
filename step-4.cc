/* ------------------------------------------------------------------------
 * 3D Space-Time Heat Equation Solver (2D Space + 1D Time) using deal.II
 * Configured with alternating Exact/Numerical output rows per cycle.
 * Exports unstructured mesh data (x, y, t, u) to a text file.
 * Includes tracking for Pointwise, Global, and Local L2 errors.
 * ------------------------------------------------------------------------ */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h> 
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h> 
#include <deal.II/lac/precondition.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>
#include <vector>
#include <string>

using namespace dealii;

// Exact solution 
template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution() : Function<dim>() {}
  virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double pi = numbers::PI;
    const double x = p[0];
    const double y = p[1];
    const double t = p[2]; // Time is the 3rd dimension

    // Peak center moves in a circle
    const double x0 = 0.5 + 0.25 * std::cos(2.0 * pi * t);
    const double y0 = 0.5 + 0.25 * std::sin(2.0 * pi * t);
    
    const double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
    return 1.0 / (1.0 + r2);
  }
};

class Step4
{
public:
  Step4();
  void run();

private:
  void make_grid(const unsigned int cycle);
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const unsigned int cycle) const;

  Triangulation<3>     triangulation; // 3D for 2D space + time
  FE_Q<3>              fe;
  DoFHandler<3>        dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs;

  ConvergenceTable     convergence_table;
};

Step4::Step4() : fe(1), dof_handler(triangulation) {}

void Step4::make_grid(const unsigned int cycle)
{
  triangulation.clear();
  // Space-time domain: x in [0,1], y in [0,1], t in [0,1]
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
  triangulation.refine_global(cycle);

  // Separate boundaries:
  // Give the t=1 boundary face an indicator of 1 so it remains unconstrained.
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->at_boundary())
        {
          for (unsigned int face_n = 0; face_n < GeometryInfo<3>::faces_per_cell; ++face_n)
            {
              if (cell->face(face_n)->at_boundary())
                {
                  // If the face center closely matches t = 1.0, reassign boundary ID
                  if (std::abs(cell->face(face_n)->center()[2] - 1.0) < 1e-10)
                    {
                      cell->face(face_n)->set_boundary_id(1);
                    }
                }
            }
        }
    }
}

void Step4::setup_system()
{
  dof_handler.distribute_dofs(fe);
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

void Step4::assemble_system()
{
  QGauss<3> quadrature_formula(fe.degree + 1);

  FEValues<3> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | 
                        update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          const Point<3> p = fe_values.quadrature_point(q_index);
          const double x = p[0];
          const double y = p[1];
          const double t = p[2];
          const double pi = numbers::PI;

          const double x0 = 0.5 + 0.25 * std::cos(2.0 * pi * t);
          const double y0 = 0.5 + 0.25 * std::sin(2.0 * pi * t);
          const double x0_dot = -0.5 * pi * std::sin(2.0 * pi * t);
          const double y0_dot = 0.5 * pi * std::cos(2.0 * pi * t);

          const double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
          const double den = 1.0 + r2;

          const double du_dt = (2.0 * (x - x0) * x0_dot + 2.0 * (y - y0) * y0_dot) / (den * den);
          const double laplacian_u = (8.0 * r2) / (den * den * den) - 4.0 / (den * den);
          const double forcing_f = du_dt - laplacian_u;

          for (const unsigned int i : fe_values.dof_indices()) 
            {
              for (const unsigned int j : fe_values.dof_indices()) 
                {
                  // Update matrix with 3D space-time gradients
                  cell_matrix(i, j) += (fe_values.shape_grad(j, q_index)[2] * fe_values.shape_value(i, q_index) + // dt term
                                        fe_values.shape_grad(j, q_index)[0] * fe_values.shape_grad(i, q_index)[0] + // dx term
                                        fe_values.shape_grad(j, q_index)[1] * fe_values.shape_grad(i, q_index)[1]) * // dy term
                                       fe_values.JxW(q_index);
                }
              cell_rhs(i) += fe_values.shape_value(i, q_index) * forcing_f * fe_values.JxW(q_index);
            }
        }

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

  // Dirichlet Boundary Conditions applied ONLY to indicator 0 (spatial boundaries + t=0)
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0, ExactSolution<3>(), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

void Step4::solve()
{
  SolverControl solver_control(10000, 1e-10);
  
  SolverGMRES<Vector<double>> solver(solver_control,
                                     SolverGMRES<Vector<double>>::AdditionalData(50));

  PreconditionJacobi<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
}

void Step4::output_results(const unsigned int cycle) const
{
  DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  
  data_out.add_data_vector(solution, "solution");
  
  ExactSolution<3> exact_solution;
  Vector<double> exact_vector(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, exact_solution, exact_vector);
  data_out.add_data_vector(exact_vector, "exact_solution");

  data_out.build_patches();
  std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
  data_out.write_vtk(output);
}

void Step4::run()
{
  // UPDATED POINTS: A = (0.25,0.25,0.25), B = (0.5,0.5,0.5), C = (0.75,0.75,0.77)
  const std::vector<Point<3>> evaluation_points = {
    Point<3>(0.25, 0.25, 0.25),
    Point<3>(0.5,  0.5,  0.5),
    Point<3>(0.75, 0.75, 0.75)
  };

  // Updated explicit column names to match new coordinates
  const std::vector<std::string> col_names = {
    "pt(0.25,0.25,0.25)", "pt(0.5,0.5,0.5)", "pt(0.75,0.75,0.75)"
  };
  
  const std::vector<std::string> suffixes = {"_A", "_B", "_C"};

  // 3D refinement expands at O(8^N). Limit cycles appropriately based on resources.
  for (unsigned int cycle = 0; cycle < 6; ++cycle) 
    {
      std::cout << "Running Cycle " << cycle << "..." << std::endl;
      make_grid(cycle);
      setup_system();
      assemble_system();
      solve();

      // Compute cell-wise differences for local integration calculations
      Vector<double> difference_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(dof_handler, solution, ExactSolution<3>(),
                                        difference_per_cell, QGauss<3>(fe.degree + 1),
                                        VectorTools::L2_norm);
      const double l2_error = VectorTools::compute_global_error(triangulation, 
                                                               difference_per_cell, 
                                                               VectorTools::L2_norm);

      ExactSolution<3> exact_solution;

      // --------------------------------------------------
      // Row A: Exact Values
      // --------------------------------------------------
      convergence_table.add_value("cycle", cycle);
      convergence_table.add_value("cells", triangulation.n_active_cells());
      convergence_table.add_value("dofs", dof_handler.n_dofs());
      convergence_table.add_value("type", std::string("exact"));
      convergence_table.add_value("global_L2_error", 0.0);

      for (unsigned int p = 0; p < evaluation_points.size(); ++p)
        {
          const double exact_val = exact_solution.value(evaluation_points[p]);
          convergence_table.add_value(col_names[p], exact_val);
          convergence_table.add_value("pt_err" + suffixes[p], 0.0);
          convergence_table.add_value("local_L2_err" + suffixes[p], 0.0);
        }

      // --------------------------------------------------
      // Row B: Numerical Values
      // --------------------------------------------------
      convergence_table.add_value("cycle", cycle);
      convergence_table.add_value("cells", triangulation.n_active_cells());
      convergence_table.add_value("dofs", dof_handler.n_dofs());
      convergence_table.add_value("type", std::string("numerical"));
      convergence_table.add_value("global_L2_error", l2_error);

      for (unsigned int p = 0; p < evaluation_points.size(); ++p)
        {
          const Point<3> &pt = evaluation_points[p];
          const double exact_val = exact_solution.value(pt);
          
          double numerical_val = 0.0;
          try 
            {
              numerical_val = VectorTools::point_value(dof_handler, solution, pt);
            }
          catch (...) 
            {
              numerical_val = exact_val; 
            }
          
          const double pointwise_error = std::abs(exact_val - numerical_val);

          double local_l2_error = 0.0;
          try 
            {
              auto cell = GridTools::find_active_cell_around_point(dof_handler, pt);
              local_l2_error = difference_per_cell(cell->active_cell_index());
            }
          catch (...) 
            {
              local_l2_error = 0.0; 
            }

          convergence_table.add_value(col_names[p], numerical_val);
          convergence_table.add_value("pt_err" + suffixes[p], pointwise_error);
          convergence_table.add_value("local_L2_err" + suffixes[p], local_l2_error);
        }
      
      output_results(cycle);

      // Export file tracking inside loop if execution reaches final cycle limit
      if (cycle == 5)
        {
          std::cout << "Exporting numerical solution grid to 'numerical_solution_final.txt'...\n";
          std::ofstream final_txt("numerical_solution_final.txt");
          final_txt << "x y t u\n"; 
          
          std::vector<Point<3>> support_points(dof_handler.n_dofs());
          DoFTools::map_dofs_to_support_points(MappingQGeneric<3>(1), dof_handler, support_points);
          
          for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
            {
              final_txt << support_points[i][0] << " "  // x
                        << support_points[i][1] << " "  // y
                        << support_points[i][2] << " "  // t
                        << solution(i) << "\n";         // u
            }
          final_txt.close();
        }
    }

  // Formatting Table Precision Definitions
  convergence_table.set_precision("global_L2_error", 4);
  convergence_table.set_scientific("global_L2_error", true);

  for (unsigned int p = 0; p < evaluation_points.size(); ++p)
    {
      convergence_table.set_precision(col_names[p], 6);
      convergence_table.set_precision("pt_err" + suffixes[p], 6);
      convergence_table.set_scientific("pt_err" + suffixes[p], true);
      convergence_table.set_precision("local_L2_err" + suffixes[p], 6);
      convergence_table.set_scientific("local_L2_err" + suffixes[p], true);
    }

  std::cout << "\n=================================== SOLUTION VALUE TABLE ===================================\n";
  convergence_table.write_text(std::cout);
  std::cout << "============================================================================================\n";
}

int main()
{
  try
    {
      Step4 space_time_heat;
      space_time_heat.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << exc.what() << std::endl;
      return 1;
    }
  return 0;
}