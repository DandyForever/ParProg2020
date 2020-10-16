#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <cmath>

double func(double x)
{
  return sin(x);
}

double square(double a, double b, double h){
	return 0.5 * (a + b) * h;
}

#define NUM 100000

double calc(double x0, double x1, double dx, uint32_t num_threads)
{
	int num_parts = (x1 - x0) / dx + 1;
	int current = 0;
	double * inputs = (double*) malloc (NUM * sizeof(double));
	double res = 0.;
	while (current < num_parts) {
		for (int i = 0; i < NUM; i++) {
			if (i + current < num_parts)
				inputs[i] = x0 + (i + current) * dx;
			else
				inputs[i] = 0;
		}
		#pragma omp parallel for reduction(+:res) num_threads(num_threads)
			for (int i = 0; i < NUM - 1; i++){
				res += square(func(inputs[i]), func(inputs[i + 1]), dx);
			}
		current += NUM - 1;
	}
	free (inputs);
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
  double x0 = 0.0, x1 =0.0, dx = 0.0;
  uint32_t num_threads = 0;
  input >> x0 >> x1 >> dx >> num_threads;

  // Calculation
  double res = calc(x0, x1, dx, num_threads);

  // Write result
  output << std::setprecision(13) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
