#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

/// @brief A structure to store the datapoint
struct sample
{
    float x;
    float y;
};

/// @brief Saves the list of samples to the given path
void saveSamples(std::vector<sample> samples, std::ofstream &path)
{
    for(sample s : samples)
    {
        path << s.x<<","<<s.y<< std::endl;
    }
}


/// @brief Generates a proposal for the next sample in the chain according to some probability g(x|y) where y is the current sample
/// @param current The current sample in the chain, y
/// @param rng The random number generator object to use
/// @param stdDev The standard deviation of the normal distribution around the current value
/// @return The proposed sample, x
sample proposeSample(sample current, std::default_random_engine &rng, float stdDev = 0.5)
{
    std::normal_distribution<float> distributionX(current.x,stdDev);
    std::normal_distribution<float> distributionY(current.y,stdDev);

    sample next;
    next.x = distributionX(rng);
    next.y = distributionY(rng);

    return next;
}

/// @brief The prior distribution to use for the samples
/// @param rng The random number generator
/// @return A sample drawn from the specified prior
sample drawFromPrior(std::default_random_engine &rng)
{
    sample priorSample;
    std::uniform_real_distribution<float> distribution(-10.0, 10.0);
    priorSample.x = distribution(rng);
    priorSample.y = distribution(rng);
    return priorSample;
}

/// @brief Gets the (un-normalized) target distribution probability of the sample f(x)
/// @param s The sample to find the probability of
float getTargetDist(sample s)
{
    float sum = s.x+s.y;
    float diff = s.x-s.y;
    float t1 = std::exp(-std::pow(sum,2.0)*0.5);
    float t2 = std::exp(-std::pow(diff,2.0));
    return t1*t2;
}

/// @brief Whether to accept the next sample based on the current one
/// @param rng The random number generator object to use
/// @return 
bool isSampleAccepted(sample next, sample current, std::default_random_engine &rng)
{
    float alpha = getTargetDist(next)/getTargetDist(current);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    float u = distribution(rng);

    return u <= alpha;
}

/// @brief A class to store and generate a markov chain using the Metropolis-Hastings algorithm
class MarkovChain
{
    private:
        unsigned int _seed;
        unsigned int _length;
        std::vector<sample> _samples;
        std::default_random_engine _rng;

    public:
        MarkovChain(unsigned int length, unsigned int seed = 0): 
        _length(length),
        // If a seed is specified, use it, otherwise use time since last epoch in ms (overflow is fine)
        _seed(seed ? seed : std::chrono::system_clock::now().time_since_epoch().count()),
        // Constructs the RNG with the specified seed
        _rng(_seed)
        {}


        void GenerateChain()
        {
            sample initialSample = drawFromPrior(_rng);
            _samples.push_back(initialSample);

            for(size_t i = 0; i<_length; i++)
            {
                sample proposal = proposeSample(_samples[i],_rng);
                sample next = isSampleAccepted(proposal,_samples[i],_rng) ? proposal : _samples[i];
                _samples.push_back(next);
            }
        }

        std::vector<sample> GetSamples(){return _samples;}
};

int main()
{
    int seed = 3;

    std::default_random_engine generator(seed);
    
    int NSamples = 1000000;
    MarkovChain mc(NSamples);

    std::cout<<"Starting chain..."<<std::endl;
    mc.GenerateChain();

    std::ofstream savePath;
    savePath.open("MCMCSamples.csv");

    saveSamples(mc.GetSamples(),savePath);

    savePath.close();

    std::cout<<"Done!"<<std::endl;

}