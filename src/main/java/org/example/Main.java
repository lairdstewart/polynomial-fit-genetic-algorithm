package org.example;

import org.apache.commons.math3.distribution.GeometricDistribution;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.Arrays;

public class Main
{
    // range -10 to 10
    private static final double[] TRUE_COEFFICIENTS = new double[]{1, -1, -4, -9, 6.5, 9.5};

    private static final RandomDataGenerator RANDOM = new RandomDataGenerator();
    private static final Parameters STANDARD_GEOMETRIC = new Parameters(10000, 50, 100, new GeometricSelector(0.01), new PointwiseRandomMutator());
    private static final Parameters STANDARD_TOP = new Parameters(10000, 50, 500, new TopOfClassSelector(),
            new PointwiseGaussianMutator());

    public static void main(String[] args)
    {
        System.out.println(Arrays.toString(geneticAlgorithm(STANDARD_TOP)));
    }

    private static double[] geneticAlgorithm(Parameters params)
    {
        Population currentPopulation = Population.generateRandomPopulation(params.populationSize);

        for (int i = 0; i < params.numIterations; i++)
        {
            Individual[] selectedIndividuals = params.selector().select(params.numSelect, currentPopulation);
            Individual[] nextGenerationIndividuals = new Individual[params.populationSize];

            for (int j = 0; j < params.populationSize; j++)
            {
                int firstIndex = RANDOM.nextInt(0, params.numSelect - 1);
                int secondIndex;
                do
                {
                    secondIndex = RANDOM.nextInt(0, params.numSelect - 1);
                } while (secondIndex == firstIndex);

                Individual selectedIndividual1 = selectedIndividuals[firstIndex];
                Individual selectedIndividual2 = selectedIndividuals[secondIndex];
                Individual child = selectedIndividual1.breed(selectedIndividual2);
                Individual mutatedChild = params.mutator().mutate(child);
                nextGenerationIndividuals[j] = mutatedChild;
            }

            currentPopulation = new Population(nextGenerationIndividuals);
            System.out.println(currentPopulation.averageFitness());
        }

        return currentPopulation.individuals[0].chromosome;
    }

    private static double evaluateFitness(double[] coefficients)
    {
        double sum = 0;
        for (double x = -2; x <= 2; x += (double) 4 / 20)
        {
            double trueValue = quinticPolynomial(x, TRUE_COEFFICIENTS);
            double individualsValue = quinticPolynomial(x, coefficients);
            sum += Math.pow(trueValue - individualsValue, 2);
        }

        return Math.pow(sum, 0.5);
    }

    /**
     * <a href="https://www.desmos.com/calculator/efxwgzdyyf">desmos</a>
     */
    private static double quinticPolynomial(double x, double[] coefs)
    {
        return coefs[0] + coefs[1] * x + coefs[2] * Math.pow(x, 2) + coefs[3] * Math.pow(x, 3) + coefs[4] * Math.pow(x, 4) + coefs[5] * Math.pow(x, 5);
    }

    interface Mutator
    {
        Individual mutate(Individual individual);
    }

    // todo could return boolean mask for performance
    interface Selector
    {
        /**
         * @param count      number of individuals to select
         * @param population population to select from
         * @return count unique selected individuals
         */
        Individual[] select(int count, Population population);
    }

    /**
     * @param populationSize
     * @param numSelect      number of individuals selected to reproduce
     */
    private record Parameters(int populationSize, int numIterations, int numSelect, Selector selector, Mutator mutator)
    {
    }

    record Individual(double[] chromosome, double fitness) implements Comparable<Individual>
    {

        Individual(double[] chromosome)
        {
            this(chromosome, evaluateFitness(chromosome));
        }

        static Individual generateRandomIndividual()
        {
            double a = Math.random() * 20 - 10;
            double b = Math.random() * 20 - 10;
            double c = Math.random() * 20 - 10;
            double d = Math.random() * 20 - 10;
            double e = Math.random() * 20 - 10;
            double f = Math.random() * 20 - 10;
            double[] genome = new double[]{a, b, c, d, e, f};

            return new Individual(genome);
        }

        Individual breed(Individual other)
        {
            double[] childCoefficients = new double[6];

            for (int i = 0; i < childCoefficients.length; i++)
            {
                if (Math.random() > 0.5)
                {
                    childCoefficients[i] = chromosome[i];
                } else
                {
                    childCoefficients[i] = other.chromosome[i];
                }
            }

            return new Individual(childCoefficients);
        }

        @Override
        public String toString()
        {
            return String.format("fitness: %f, chromosome: %s", fitness, Arrays.toString(chromosome));
        }

        @Override
        public int compareTo(Individual o)
        {
            double compare = fitness - o.fitness;
            if (compare < 0) return -1;
            else if (compare > 0) return 1;
            else return 0;
        }
    }

    static class WeightedSelector implements Selector
    {
        // todo
        @Override
        public Individual[] select(int count, Population population)
        {
            return new Individual[0];
        }
    }

    static class TopOfClassSelector implements Selector
    {
        @Override
        public Individual[] select(int count, Population population)
        {
            return Arrays.copyOfRange(population.individuals, 0, count);
        }

    }

    static class GeometricSelector implements Selector
    {
        private final GeometricDistribution geometricDistribution;

        GeometricSelector(double p)
        {
            geometricDistribution = new GeometricDistribution(p);
        }

        @Override
        public Individual[] select(int count, Population population)
        {
            boolean[] selectedIndices = new boolean[population.size];
            Individual[] selectedIndividuals = new Individual[count];

            int numSelected = 0;
            while (numSelected < count)
            {
                int sampleIndex = sample(population.size);

                while (sampleIndex < population.size)
                {
                    if (!selectedIndices[sampleIndex])
                    {
                        selectedIndices[sampleIndex] = true;
                        selectedIndividuals[numSelected] = population.individuals[sampleIndex];
                        numSelected++;
                        break;
                    }

                    // if already selected, select the next individual and so on
                    sampleIndex++;
                }
            }

            return selectedIndividuals;
        }

        int sample(int upperBoundExclusive)
        {
            int sample = geometricDistribution.inverseCumulativeProbability(Math.random());
            if (sample >= upperBoundExclusive)
            {
                return sample(upperBoundExclusive);
            }

            return sample;
        }
    }

    record Population(Individual[] individuals, int size)
    {
        Population(Individual[] individuals)
        {
            this(individuals, individuals.length);
            Arrays.sort(individuals);
        }

        static Population generateRandomPopulation(int size)
        {
            Individual[] individuals = new Individual[size];
            for (int i = 0; i < size; i++)
            {
                individuals[i] = Individual.generateRandomIndividual();
            }

            return new Population(individuals);
        }

        @Override
        public String toString()
        {
            StringBuilder builder = new StringBuilder();

            for (Individual individual : individuals)
            {
                builder.append(individual.toString());
                builder.append("\n");
            }

            return builder.toString();
        }

        double averageFitness()
        {
            double sum = 0;
            for (Individual individual : individuals)
            {
                sum += individual.fitness;
            }
            return sum / individuals.length;
        }
    }

    static class PointwiseGaussianMutator implements Mutator
    {
        @Override
        public Individual mutate(Individual individual)
        {
            if (RANDOM.nextUniform(0, 1) < 0.5)
            {
                double[] coefficients = individual.chromosome();
                int randomCoefficientIndex = RANDOM.nextInt(0, 5);
                coefficients[randomCoefficientIndex] = RANDOM.nextGaussian(coefficients[randomCoefficientIndex], 1);
                return new Individual(coefficients);
            }

            return individual;
        }
    }

    static class PointwiseRandomMutator implements Mutator
    {
        @Override
        public Individual mutate(Individual individual)
        {
            if (RANDOM.nextUniform(0, 1) < 0.5)
            {
                double[] coefficients = individual.chromosome();
                int randomCoefficientIndex = RANDOM.nextInt(0, 5);
                coefficients[randomCoefficientIndex] = RANDOM.nextUniform(-10, 10);
                return new Individual(coefficients);
            }

            return individual;
        }
    }
}
