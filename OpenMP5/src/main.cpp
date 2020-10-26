#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <vector>

void calc(uint32_t xSize, uint32_t ySize, uint32_t iterations, uint32_t num_threads, uint8_t* inputFrame, uint8_t* outputFrame)
{
	std::vector<std::vector<std::vector<char>>> frame(2);
	for (uint32_t i = 0; i < xSize; i++) {
		frame[0].push_back(std::vector<char>(ySize));
		frame[1].push_back(std::vector<char>(ySize));
	}

	for (uint32_t y = 0; y < ySize; y++) {
		for (uint32_t x = 0; x < xSize; x++) {
			frame[0][x][y] = (int)inputFrame[y * xSize + x];
		}
	}


	int prev = num_threads - num_threads;
	int curr = 0;
	uint32_t block_size = (xSize + num_threads - 1) / num_threads;
	#pragma omp parallel num_threads(num_threads)
	{
		uint32_t ind = omp_get_thread_num();
		uint32_t first = ind * block_size;
		uint32_t last = (ind + 1) * block_size;
		if (last > xSize)
			last = xSize;

		for (uint32_t i = 0; i < iterations; i++) {
			#pragma omp barrier
			prev = i % 2;
			curr = (i + 1) % 2;
			for (uint32_t x = first; x < last; x++) {
				for (uint32_t y = 0; y < ySize; y++) {
					int count = 0;
					int x1 = x == 0 ? xSize - 1 : x - 1;
					int x2 = x == xSize - 1 ? 0 : x + 1;
					int y1 = y == 0 ? ySize - 1 : y - 1;
					int y2 = y == ySize - 1 ? 0 : y + 1;
					if (frame[prev][x1][y1] == 1) count++;
					if (frame[prev][x][y1] == 1) count++;
					if (frame[prev][x1][y] == 1) count++;
					if (frame[prev][x2][y1] == 1) count++;
					if (frame[prev][x1][y2] == 1) count++;
					if (frame[prev][x2][y2] == 1) count++;
					if (frame[prev][x][y2] == 1) count++;
					if (frame[prev][x2][y] == 1) count++;
					if (frame[prev][x][y] == 0 && count == 3) frame[curr][x][y] = 1;
					else if (frame[prev][x][y] == 1 && (count < 2 || count > 3)) frame[curr][x][y] = 0;
					else frame[curr][x][y] = frame[prev][x][y];
				}
			}
		}
	}

	for (uint32_t y = 0; y < ySize; y++) {
		for (uint32_t x = 0; x < xSize; x++) {
			outputFrame[y * xSize + x] = frame[curr][x][y];
		}
	}
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
  uint32_t xSize = 0, ySize = 0, iterations = 0, num_threads = 0;
  input >> xSize >> ySize >> iterations >> num_threads;
  uint8_t* inputFrame = new uint8_t[xSize*ySize];
  uint8_t* outputFrame = new uint8_t[xSize*ySize];
  for (uint32_t y = 0; y < ySize; y++)
  {
    for (uint32_t x = 0; x < xSize; x++)
    {
      input >> inputFrame[y*xSize + x];
      inputFrame[y*xSize + x] -= '0';
    }
  }

  // Calculation
  calc(xSize, ySize, iterations, num_threads, inputFrame, outputFrame);

  // Write result
  for (uint32_t y = 0; y < ySize; y++)
  {
    for (uint32_t x = 0; x < xSize; x++)
    {
      outputFrame[y*xSize + x] += '0';
      output << " " << outputFrame[y*xSize + x];
    }
    output << "\n";
  }

  // Prepare to exit
  delete outputFrame;
  delete inputFrame;
  output.close();
  input.close();
  return 0;
}
