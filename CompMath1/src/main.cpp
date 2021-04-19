#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

double acceleration(double t)
{
  return sin(t);
}

void calc(double* trace, uint32_t traceSize, double t0, double dt, double y0, double y1, int rank, int size)
{
    MPI_Bcast(&traceSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    uint32_t block_size = (traceSize + size) / size;
    uint32_t start = rank * block_size;
    uint32_t finish = rank * block_size + block_size;
    finish = (finish > traceSize) ? traceSize : finish;

    if (rank != 0) {
        trace = new double[finish - start];
    }

    trace[0] = 0.;
    trace[1] = 0.;
    if (rank == 0) {
        trace[0] = y0;
        trace[1] = y0;
    }
    for (uint32_t i = start + 2; i < finish; i++) {
        uint32_t index = i - start;
        trace[index] = dt * dt * acceleration(t0 + (i - 1) * dt) + 2 * trace[index - 1] - trace[index - 2];
    }
    
    uint32_t calc_size = finish - start;
    double* calculated;
    int* blocks_size, *shifts;

    if (rank == 0) {
        blocks_size = new int[size];
        shifts = new int[size];

        for (int i = 0; i < size; i++) {
            start = i * block_size;
            finish = i * block_size + block_size;
            finish = (finish > traceSize) ? traceSize : finish;
            blocks_size[i] = finish - start;
            shifts[i] = start;
        }
        
        calculated = new double[traceSize];

        MPI_Gatherv(trace, calc_size, MPI_DOUBLE, calculated, blocks_size, shifts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        

        double b = 0.;
        double u = 0.;
        double v = 0.;
        double a = y0;
        double a_add = 0.;
        double v_add = 0.;
        uint32_t ft = 0, lt = 0;
        for (int i = 1; i < size; i++) {
            u = (trace[shifts[i] - 1] - trace[shifts[i] - 2]);
            b = trace[shifts[i] - 1];
            a_add = a + b + v * (blocks_size[i - 1]);
            v_add = u + v;
            ft = shifts[i];
            if (i == size - 1)
                lt = traceSize;
            else
                lt = shifts[i + 1];

            for (uint32_t j = ft; j < lt; j++) {
                uint32_t param = j - ft;
                trace[j] = calculated[j] + a_add + param * v_add;
            }

            a = trace[lt - 1];
            v = u;
        }

        u = (y1 - trace[traceSize - 1]) / (traceSize);
        a = 0.;
        b = 0.;
        v = 0.;
        a_add = 0.;
        v_add = u;

        for (int i = 0; i < size; i++) {
            ft = shifts[i];
            if (i == size - 1)
                lt = traceSize;
            else
                lt = shifts[i + 1];

            for (uint32_t j = ft; j < lt; j++) {
                uint32_t param = j - ft;
                trace[j] = calculated[j] + a_add + param * v_add;
            }

            a = trace[ft];
            v = u;
            u = (trace[lt - 1] - trace[lt - 2]);
            b = trace[lt - 1];
            a_add = a + b + v * (blocks_size[i]);
            v_add = u + v;
        }

        delete [] calculated;
        delete [] shifts;
        delete [] blocks_size;

    } else {
        MPI_Gatherv(trace, calc_size, MPI_DOUBLE, calculated, blocks_size, shifts, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        delete [] trace;
    }
/*
        double v = 0.;
        double a = y0;
        double u = trace[calc_size - 1] / (calc_size * dt);
        double b = trace[calc_size - 1];
        
        MPI_Send(&u, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&v, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&a, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&b, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

        double b_ = 0.;
        MPI_Status status;
        MPI_Recv(&b_, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, &status);
        v0 = (y1 - b_) / (dt * traceSize);


  // Sighting shot
  double v0 = 0;
  if (rank == 0 && size > 0)
  {
    trace[0] = y0;
    trace[1] = y0 + dt*v0;
    for (uint32_t i = 2; i < traceSize; i++)
    {
      trace[i] = dt*dt*acceleration(t0 + (i - 1)*dt) + 2*trace[i - 1] - trace[i - 2];
    }
  }

  // The final shot
  if (rank == 0 && size > 0)
  {
    v0 = (y1 - trace[traceSize - 1])/(dt*traceSize);
    trace[0] = y0;
    trace[1] = y0 + dt*v0;
    for (uint32_t i = 2; i < traceSize; i++)
    {
      trace[i] = dt*dt*acceleration(t0 + (i - 1)*dt) + 2*trace[i - 1] - trace[i - 2];
    }
  }
*/
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  uint32_t traceSize = 0;
  double t0 = 0, t1 = 0, dt = 0, y0 = 0, y1 = 0;
  double* trace = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> t0 >> t1 >> dt >> y0 >> y1;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    traceSize = (t1 - t0)/dt;
    trace = new double[traceSize];

    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(trace, traceSize, t0, dt, y0, y1, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete trace;
      return 1;
    }

    for (uint32_t i = 0; i < traceSize; i++)
    {
      output << " " << trace[i];
    }
    output << std::endl;
    output.close();
    delete trace;
  }

  MPI_Finalize();
  return 0;
}
