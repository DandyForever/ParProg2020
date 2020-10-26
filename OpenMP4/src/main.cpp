#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <vector>

double calc(uint32_t x_last, uint32_t num_threads) {
	std::vector <std::pair<double, double>> ans(num_threads);
	uint32_t block_size = x_last / num_threads;
	if (x_last % num_threads != 0)
		block_size++;
	x_last--;

	#pragma omp parallel num_threads(num_threads)
	{
		uint32_t ind = omp_get_thread_num();
		uint32_t first = block_size * ind + 1;
		uint32_t last = block_size * (ind + 1);
		if (last > x_last)
			last = x_last;

		double res = 0.;
		double current = 1.;
		for (uint32_t i = first; i <= last; i++) {
			current /= i;
			res += current;
		}
		ans[ind] = std::make_pair(res, current);
	}

	double res = 1.;
	double current = 1.;
	for (uint32_t i = 0; i < num_threads; i++) {
		res += current * ans[i].first;
		current *= ans[i].second;
	}

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
  output << std::setprecision(16) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
