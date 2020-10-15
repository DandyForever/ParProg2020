#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

#define NUM 100000

double calc(uint32_t x_last, uint32_t num_threads)
{
  double* sums = (double*) malloc(NUM * sizeof(double));
	double res = 0;
	uint32_t current = 0;
	while (current < x_last) {
  	#pragma omp parallel for num_threads(num_threads)
		for (int i = 0; i < NUM; i++)
			if (x_last - current - i > 0 && x_last - current - i <= x_last)
				sums[i] = 1. / (x_last - current - i);
			else 
				sums[i] = 0;

  	for (uint32_t i = 0; i < NUM; i++)
			res += sums[i];
		current += NUM;
	}
  free(sums);

  return res;
}

int main(int argc, char** argv)
{
  // Check arguments
  if (argc != 3)
  {
    std::cout << "[Error] Usage <inputfile> <output file>\n";
    return 1;
  }

  // Prepare input file
  std::ifstream input(argv[1]);
  if (!input.is_open())
  {
    std::cout << "[Error] Can't open " << argv[1] << " for write\n";
    return 1;
  }

  // Prepare output file
  std::ofstream output(argv[2]);
  if (!output.is_open())
  {
    std::cout << "[Error] Can't open " << argv[2] << " for read\n";
    input.close();
    return 1;
  }

  // Read arguments from input
  uint32_t x_last = 0, num_threads = 0;
  input >> x_last >> num_threads;
  // Calculation
  double res = calc(x_last, num_threads);

  // Write result
  output << std::setprecision(15) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
