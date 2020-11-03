#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

void calc(double* arr, uint32_t zSize, uint32_t ySize, uint32_t xSize, int rank, int size)
{
  if (rank == 0 && size > 0) {
		double * tasks = (double*) malloc (xSize * ySize * zSize * sizeof(double));
		uint32_t * task_size = (uint32_t *) malloc (xSize * ySize * zSize * sizeof(uint32_t));
		uint32_t * flags = (uint32_t *) calloc (xSize * ySize * zSize, sizeof(uint32_t));
		uint32_t current = 0, current_num = 0, index = 0, zCpy = 0;
		int yCpy = 0, xCpy = 0;
		for (uint32_t z = 0; z < zSize; z++) {
			for (uint32_t y = 0; y < ySize; y++) {
				for (uint32_t x = 0; x < xSize; x++) {
					index = z * ySize * xSize + y * ySize + x;
					if (!flags[index]) {
						zCpy = z; yCpy = y; xCpy = x;
						while (zCpy < zSize && yCpy >= 0 && xCpy >= 0) {
							flags[index] = 1;
							tasks[current++] = arr[index];
							zCpy++; yCpy--; xCpy--;
							index = zCpy * ySize * xSize + yCpy * ySize + xCpy;
						}
						task_size[current_num++] = current;
					}
				}
			}
		}

		uint32_t block_size = (current + size) / size;
		uint32_t current_proc = 0, current_index = 0, current_task = 0, last_guy = 0;
		uint32_t task_number[10], task_weight[10];
		for (uint32_t i = 0; i < current_num; i++) {
			task_size[i] -= last_guy;
			if (task_size[i] > block_size || i == current_num - 1) {
				task_number[current_proc] = i - current_index;
				task_weight[current_proc] = task_size[i];
				current_index = i;
				last_guy += task_size[i];
				current_proc++;
			}
		}

		for (int i = 1; i < size; i++) {
			MPI_Send(&task_number[i], 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
			MPI_Send(&task_weight[i], 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
		}

		current_index = task_number[0], current_task = task_weight[0];
		for (int i = 1; i < size; i++) {
			MPI_Send(&tasks[current_task], task_weight[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&task_size[current_index + 1], task_number[i], MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
			current_task += task_weight[i];
			current_index += task_number[i];
		}

		current_index = 0;
		for (uint32_t i = 1; i < task_weight[0]; i++) {
			if (i != task_size[current_index])
				tasks[i] = sin(tasks[i - 1]);
			else
				current_index++;
		}

		current_index = task_number[0]; current_task = task_weight[0];
		MPI_Status status;
		for (int i = 1; i < size; i++) {
			MPI_Recv(&tasks[current_task], task_weight[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
			current_task += task_weight[i];
		}

		current = 0;
		for (uint32_t z = 0; z < zSize; z++) {
			for (uint32_t y = 0; y < ySize; y++) {
				for (uint32_t x = 0; x < xSize; x++) {
					index = z * ySize * xSize + y * ySize + x;
					if (flags[index]) {
						zCpy = z; yCpy = y; xCpy = x;
						while (zCpy < zSize && yCpy >= 0 && xCpy >= 0) {
							flags[index] = 0;
							arr[index] = tasks[current++];
							zCpy++; yCpy--; xCpy--;
							index = zCpy * ySize * xSize + yCpy * ySize + xCpy;
						}
					}
				}
			}
		}

		free(tasks); free(task_size); free(flags);
  } else {
		uint32_t task_number = 0, task_weight = 0;
		MPI_Status status;
		MPI_Recv(&task_number, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&task_weight, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
		
		double* local_task = (double*) malloc (task_weight * sizeof(double));
		uint32_t* local_task_size = (uint32_t*) malloc (task_number * sizeof(uint32_t));
		MPI_Recv(local_task, task_weight, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(local_task_size, task_number, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);

		uint32_t current_index = 0;
		for (uint32_t i = 1; i < task_weight; i++) {
			if (i != local_task_size[current_index])
				local_task[i] = sin(local_task[i - 1]);
			else
				current_index++;
		}

		MPI_Send(local_task, task_weight, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		free(local_task); free(local_task_size);
	}
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, buf = 0;
  uint32_t zSize = 0, ySize = 0, xSize = 0;
  double* arr = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> zSize >> ySize >> xSize;
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    arr = new double[zSize * ySize * xSize];
    for (uint32_t z = 0; z < zSize; z++) {
      for (uint32_t y = 0; y < ySize; y++) {
        for (uint32_t x = 0; x < xSize; x++) {
          input >> arr[z*ySize*xSize + y*xSize + x];
        }
      }
    }
    input.close();
  } else {
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (buf != 0)
    {
      return 1;
    }
  }

  calc(arr, zSize, ySize, xSize, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete arr;
      return 1;
    }

    for (uint32_t z = 0; z < zSize; z++) {
      for (uint32_t y = 0; y < ySize; y++) {
        for (uint32_t x = 0; x < xSize; x++) {
          output << " " << arr[z*ySize*xSize + y*xSize + x];
        }
        output << std::endl;
      }
      output << std::endl;
    }
    output.close();
    delete arr;
  }

  MPI_Finalize();
  return 0;
}
