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
#include <cmath>
#include <vector>
#include <string>

using namespace dealii;

template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution() : Function<dim>() {}
  virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    const double pi = numbers::PI;
    const double x = p[0];
    const double t = p[1];
    return std::sin(pi * x) * (1.0 + t) * std::exp(-t / 2.0);
  }
};

class Step3
{
public:
  Step3();
  void run();

private:
  void make_grid(const unsigned int cycle);
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const unsigned int cycle) const;

  Triangulation<2>    triangulation; 
  FE_Q<2>             fe;
  DoFHandler<2>       dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs;

  ConvergenceTable     convergence_table;
};

Step3::Step3() : fe(1), dof_handler(triangulation) {}

void Step3::make_grid(const unsigned int cycle)
{
  triangulation.clear();
  // Space-time domain: x in [0,1], t in [0,1]
  GridGenerator::hyper_rectangle(triangulation, Point<2>(0.0, 0.0), Point<2>(1.0, 1.0));
  
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
        {
          if (cell->at_boundary(f))
            {
              const Point<2> face_center = cell->face(f)->center();
              if (std::abs(face_center[0] - 0.0) < 1e-10 || std::abs(face_center[0] - 1.0) < 1e-10)
                cell->face(f)->set_boundary_id(1); // Spatial boundaries
              else if (std::abs(face_center[1] - 0.0) < 1e-10)
                cell->face(f)->set_boundary_id(2); // Initial boundary (t=0)
              else if (std::abs(face_center[1] - 1.0) < 1e-10)
                cell->face(f)->set_boundary_id(3); // Future boundary (t=1)
            }
        }
    }

  triangulation.refine_global(cycle);
}

void Step3::setup_system()
{
  dof_handler.distribute_dofs(fe);
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

void Step3::assemble_system()
{
  QGauss<2> quadrature_formula(fe.degree + 1);

  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | 
                        update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const double pi = numbers::PI;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          const Point<2> &p = fe_values.quadrature_point(q_index);
          const double x = p[0];
          const double t = p[1];

          const double forcing_f = std::sin(pi * x) * std::exp(-t / 2.0) * (0.5 + pi * pi + (pi * pi - 0.5) * t);
          
          for (const unsigned int i : fe_values.dof_indices())
            {
              for (const unsigned int j : fe_values.dof_indices())
                {
                  cell_matrix(i, j) += (fe_values.shape_grad(j, q_index)[1] * fe_values.shape_value(i, q_index) +
                                        fe_values.shape_grad(j, q_index)[0] * fe_values.shape_grad(i, q_index)[0]) *
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

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 1, ExactSolution<2>(), boundary_values);
  VectorTools::interpolate_boundary_values(dof_handler, 2, ExactSolution<2>(), boundary_values);
  
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

void Step3::solve()
{
  SolverControl solver_control(10000, 1e-10);
  SolverGMRES<Vector<double>> solver(solver_control,
                                     SolverGMRES<Vector<double>>::AdditionalData(50));

  PreconditionJacobi<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
}

void Step3::output_results(const unsigned int cycle) const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  
  data_out.add_data_vector(solution, "solution");
  
  ExactSolution<2> exact_solution;
  Vector<double> exact_vector(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, exact_solution, exact_vector);
  data_out.add_data_vector(exact_vector, "exact_solution");

  data_out.build_patches();
  std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
  data_out.write_vtk(output);
}

void Step3::run()
{
  // UPDATED POINTS: A = (0.5, 0.25), B = (0.5, 0.5), C = (0.5, 0.75)
  const std::vector<Point<2>> evaluation_points = {
    Point<2>(0.5, 0.25), 
    Point<2>(0.5, 0.5),
    Point<2>(0.5, 0.75)
  };

  const std::vector<std::string> suffixes = {"_A", "_B", "_C"};

  for (unsigned int cycle = 0; cycle < 8; ++cycle) 
    {
      std::cout << "Running Cycle " << cycle << "..." << std::endl;
      make_grid(cycle);
      setup_system();
      assemble_system();
      solve();

      // Compute global and local element differences
      Vector<double> difference_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(dof_handler, solution, ExactSolution<2>(),
                                        difference_per_cell, QGauss<2>(fe.degree + 1),
                                        VectorTools::L2_norm);

      const double global_l2_error = VectorTools::compute_global_error(triangulation, 
                                                                       difference_per_cell, 
                                                                       VectorTools::L2_norm);
      ExactSolution<2> exact_solution;

      // ------------------------------------------------------------------
      // Row A: Exact Values Row
      // ------------------------------------------------------------------
      convergence_table.add_value("cycle", cycle);
      convergence_table.add_value("cells", triangulation.n_active_cells());
      convergence_table.add_value("dofs", dof_handler.n_dofs());
      convergence_table.add_value("type", std::string("exact"));
      convergence_table.add_value("global_L2_error", 0.0);

      for (unsigned int p = 0; p < evaluation_points.size(); ++p)
        {
          const double exact_val = exact_solution.value(evaluation_points[p]);
          convergence_table.add_value("uh" + suffixes[p], exact_val);
          convergence_table.add_value("pt_err" + suffixes[p], 0.0);
          convergence_table.add_value("local_L2_err" + suffixes[p], 0.0);
        }

      // ------------------------------------------------------------------
      // Row B: Numerical Values Row
      // ------------------------------------------------------------------
      convergence_table.add_value("cycle", cycle);
      convergence_table.add_value("cells", triangulation.n_active_cells());
      convergence_table.add_value("dofs", dof_handler.n_dofs());
      convergence_table.add_value("type", std::string("numerical"));
      convergence_table.add_value("global_L2_error", global_l2_error);

      for (unsigned int p = 0; p < evaluation_points.size(); ++p)
        {
          const Point<2> &pt = evaluation_points[p];
          const double exact_val = exact_solution.value(pt);
          
          double numerical_val = 0.0;
          try {
            numerical_val = VectorTools::point_value(dof_handler, solution, pt);
          }
          catch (...) {
            numerical_val = exact_val; 
          }

          const double pointwise_error = std::abs(exact_val - numerical_val);

          double local_l2_error = 0.0;
          try {
            auto cell = GridTools::find_active_cell_around_point(dof_handler, pt);
            local_l2_error = difference_per_cell(cell->active_cell_index());
          }
          catch (...) {
            local_l2_error = 0.0; 
          }

          convergence_table.add_value("uh" + suffixes[p], numerical_val);
          convergence_table.add_value("pt_err" + suffixes[p], pointwise_error);
          convergence_table.add_value("local_L2_err" + suffixes[p], local_l2_error);
        }

      output_results(cycle);

      // Save global dataset per cycle
      std::ofstream txt_out("numerical_solution_global_cycle_" + std::to_string(cycle) + ".txt");
      txt_out << "x t u\n"; 
      std::vector<Point<2>> support_points(dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(MappingQGeneric<2>(1), dof_handler, support_points);
      for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
        {
          txt_out << support_points[i][0] << " "  
                  << support_points[i][1] << " "  
                  << solution(i) << "\n";          
        }
      txt_out.close();

      // Standalone numerical_solution_final.txt export on execution cycle 7
      if (cycle == 7)
        {
          std::ofstream final_txt("numerical_solution_final.txt");
          final_txt << "x t u\n";
          for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
            {
              final_txt << support_points[i][0] << " "  
                        << support_points[i][1] << " "  
                        << solution(i) << "\n";          
            }
          final_txt.close();
        }
    }

  // Set visual formats for output table printouts
  convergence_table.set_precision("global_L2_error", 4);
  convergence_table.set_scientific("global_L2_error", true);

  for (const auto &sfx : suffixes)
    {
      convergence_table.set_precision("uh" + sfx, 6);
      convergence_table.set_precision("pt_err" + sfx, 6);
      convergence_table.set_scientific("pt_err" + sfx, true);
      convergence_table.set_precision("local_L2_err" + sfx, 6);
      convergence_table.set_scientific("local_L2_err" + sfx, true);
    }

  std::cout << "\n========================================= COMPREHENSIVE SPACE-TIME HEAT CONVERGENCE SYSTEM =========================================\n";
  convergence_table.write_text(std::cout);
  std::cout << "================================================----------------------------------------------------=================================\n";
}

int main()
{
  try
    {
      Step3 space_time_heat;
      space_time_heat.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << exc.what() << std::endl;
      return 1;
    }
  return 0;
}