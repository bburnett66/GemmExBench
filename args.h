
#include <string>
#include <iostream>

struct Args {
	int m, k, n, n_runs=100;
	bool is_copy=false;
};

bool get_args(int argc, char* argv[], Args* a) 
{
	if (argc < 7)
	{
		std::cerr << "Not enough arguments" << std::endl;
		std::cerr << "Usage:" << std::endl
			<< "./gemmEx --m M --k K --n N {--n-runs 100 --copy}" << std::endl;

		return false;
	}
	for (int i = 1; i < argc; i += 2)
	{
		if (std::string(argv[i]) == "--m")
			a->m = std::stoi(std::string(argv[i+1]));
		else if (std::string(argv[i]) == "--k")
			a->k = std::stoi(std::string(argv[i+1]));
		else if (std::string(argv[i]) == "--n")
			a->n = std::stoi(std::string(argv[i+1]));
		else if (std::string(argv[i]) == "--n-runs")
			a->n_runs = std::stoi(std::string(argv[i+1]));
		else if (std::string(argv[i]) == "--copy")
			a->is_copy = true;
		else
		{
			std::cerr << "Unrecognized argument: " 
				<< std::string(argv[i]) << std::endl
				<< "exiting..." << std::endl;
			return false;
		}
	}
	return true;
}