package org.example;

import org.apache.commons.math3.distribution.GeometricDistribution;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * smaller Params.numSelect makes convergence faster and smoother (~1/100). As it gets larger (~1/10) convergence is
 * slower and more jagged. The benefit of the latter is not getting stuck in local minima, but I'm not sure how much
 * of an issue that is here.
 * <p>
 * larger population size does help, of course makes things much slower. Here 1k -> 10k gives 2x smaller loss. Seeing
 * not much improvement after 10k. Convergence gets smoother.
 * <p>
 * Hard to tell any difference between Geometric selector and top of class. Geometric convergence is a bit jaggeder.
 * <p>
 * The biggest impact is the choice of mutation method by far. Moving from a totally random mutation to a gaussian
 * jiggle gave an immediate 2x improvement.
 */
public class Main
{
    private static int NUM_FITNESS_EVALUATIONS = 0;

    // todo curious to try and see what optimal setup there is when we only have 1000 sample size
    // range -10 to 10
    private static final double[] TRUE_COEFFICIENTS = new double[]{1, -1, -4, -9, 6.5, 9.5};

    private static final RandomDataGenerator RANDOM = new RandomDataGenerator();
    private static final Parameters GEOMETRIC_POINTWISE = new Parameters(10000, 50, new GeometricSelector(0.01,
            0.1), new PointwiseRandomMutator(), new PointwiseBreeder()); // 80
    private static final Parameters TOP_GAUSSIAN_MID_SIGMA = new Parameters(10000, 50, new TopOfClassSelector(0.1),
            new PointwiseGaussianMutator(1), new PointwiseBreeder()); // 8
    private static final Parameters TOP_GAUSSIAN_SMALL_SIGMA = new Parameters(10000, 50,
            new TopOfClassSelector(0.1), new PointwiseGaussianMutator(0.1), new PointwiseBreeder()); // 1
    private static final Parameters TOP_NO_MUTATION = new Parameters(10000, 50, new TopOfClassSelector(0.1),
            new NullMutator(), new PointwiseBreeder()); // 1
    private static final Parameters DO_NOTHING = new Parameters(10000, 50, new TopOfClassSelector(0.1),
            new NullMutator(), new NullBreeder());

    public static void main(String[] args)
    {
        List<double[]> chromosomeHistory = geneticAlgorithm(TOP_NO_MUTATION);
        saveToDat(chromosomeHistory);
        System.out.println("total polynomial evaluations: " + NUM_FITNESS_EVALUATIONS);
    }

    private static void saveToDat(List<double[]> arrays)
    {
        String fileName = "chromosome-history.dat";

        try (FileWriter fileWriter = new FileWriter(fileName);
             BufferedWriter bufferedWriter = new BufferedWriter(fileWriter))
        {
            for (double[] array : arrays)
            {
                for (int i = 0; i < array.length - 1; i++)
                {
                    bufferedWriter.write(array[i] + ",");
                }

                bufferedWriter.write(array[array.length - 1] + "");
                bufferedWriter.newLine();
            }
        } catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    /**
     * 1. kill off all parents and only keep children for the next generation
     * 2. create 80 children and keep 20 parents for the next generation
     * 3. create 500 children, down-select to 100
     */
    private static List<double[]> geneticAlgorithm(Parameters params)
    {
        List<double[]> chromosomeHistory = new ArrayList<>();

        Population currentPopulation = Population.generateRandomPopulation(params.populationSize);

        for (int i = 0; i < params.numIterations; i++)
        {
            chromosomeHistory.add(currentPopulation.individuals[0].chromosome);
            System.out.println(currentPopulation.averageFitness());

            Individual[] selectedIndividuals = params.selector().select(currentPopulation);
            Individual[] nextGenerationIndividuals = new Individual[params.populationSize];

            for (int j = 0; j < params.populationSize; j++)
            {
                int firstIndex = RANDOM.nextInt(0, selectedIndividuals.length - 1);
                int secondIndex;
                do
                {
                    secondIndex = RANDOM.nextInt(0, selectedIndividuals.length - 1);
                } while (secondIndex == firstIndex);

                Individual selectedIndividual1 = selectedIndividuals[firstIndex];
                Individual selectedIndividual2 = selectedIndividuals[secondIndex];
                Individual child = params.breeder.breed(selectedIndividual1, selectedIndividual2);
                Individual mutatedChild = params.mutator().mutate(child);
                nextGenerationIndividuals[j] = mutatedChild;
            }

            Population nextPopulation = new Population(nextGenerationIndividuals);

            if (currentPopulation.averageFitness() == nextPopulation.averageFitness())
            {
                break;
            }

            currentPopulation = nextPopulation;
        }

        return chromosomeHistory;
    }

    private static double evaluateFitness(double[] coefficients)
    {
        NUM_FITNESS_EVALUATIONS++;

        double sum = 0;
        for (double x = -2; x <= 2; x += (double) 4 / 20)
        {
            double trueValue = quinticPolynomial(x, TRUE_COEFFICIENTS);
            double individualsValue = quinticPolynomial(x, coefficients);
            sum += Math.pow(trueValue - individualsValue, 4); // better than quadraditc
        }

        return Math.pow(sum, 0.25);
    }

    /**
     * <a href="https://www.desmos.com/calculator/efxwgzdyyf">desmos</a>
     */
    private static double quinticPolynomial(double x, double[] coefs)
    {
        return coefs[0] + coefs[1] * x + coefs[2] * Math.pow(x, 2) + coefs[3] * Math.pow(x, 3) + coefs[4] * Math.pow(x, 4) + coefs[5] * Math.pow(x, 5);
    }

    static class SinglePointCrossover implements Breeder
    {
        @Override
        public Individual breed(Individual first, Individual second)
        {
            double[] childCoefficients = new double[6];

            int crossoverIndex = RANDOM.nextInt(0, 5);

            for (int i = 0; i < 6; i++)
            {
                if (i < crossoverIndex)
                {
                    childCoefficients[i] = first.chromosome[i];
                } else
                {
                    childCoefficients[i] = second.chromosome[i];
                }
            }

            return new Individual(childCoefficients);
        }
    }

    static class PointwiseBreeder implements Breeder
    {
        @Override
        public Individual breed(Individual first, Individual second)
        {
            double[] childCoefficients = new double[6];

            for (int i = 0; i < childCoefficients.length; i++)
            {
                if (RANDOM.nextUniform(0, 1) > 0.5)
                {
                    childCoefficients[i] = first.chromosome[i];
                } else
                {
                    childCoefficients[i] = second.chromosome[i];
                }
            }

            return new Individual(childCoefficients);
        }
    }

    interface Breeder
    {
        Individual breed(Individual first, Individual second);
    }

    interface Mutator
    {
        Individual mutate(Individual individual);
    }

    interface Selector
    {
        Individual[] select(Population population);
    }

    private record Parameters(int populationSize, int numIterations, Selector selector, Mutator mutator,
                              Breeder breeder)
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
        public Individual[] select(Population population)
        {
            return new Individual[0];
        }
    }

    static class TopOfClassSelector implements Selector
    {
        final double pctToSelect;

        TopOfClassSelector(double pctToSelect)
        {
            this.pctToSelect = pctToSelect;
        }

        @Override
        public Individual[] select(Population population)
        {
            int numToSelect = (int) (population.size * pctToSelect);

            return Arrays.copyOfRange(population.individuals, 0, numToSelect);
        }

    }

    static class GeometricSelector implements Selector
    {
        private final GeometricDistribution geometricDistribution;
        private final double pctToSelect;

        GeometricSelector(double p, double pctToSelect)
        {
            geometricDistribution = new GeometricDistribution(p);
            this.pctToSelect = pctToSelect;
        }

        @Override
        public Individual[] select(Population population)
        {
            int numToSelect = (int) (population.size * pctToSelect);

            boolean[] selectedIndices = new boolean[population.size];
            Individual[] selectedIndividuals = new Individual[numToSelect];

            int numSelected = 0;
            while (numSelected < numToSelect)
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

    static class NullMutator implements Mutator
    {
        @Override
        public Individual mutate(Individual individual)
        {
            return individual;
        }
    }

    static class PointwiseGaussianMutator implements Mutator
    {
        private final double sigma;

        PointwiseGaussianMutator(double sigma)
        {
            this.sigma = sigma;
        }

        @Override
        public Individual mutate(Individual individual)
        {
            if (RANDOM.nextUniform(0, 1) < 0.5)
            {
                double[] coefficients = individual.chromosome();
                int randomCoefficientIndex = RANDOM.nextInt(0, 5);
                coefficients[randomCoefficientIndex] = RANDOM.nextGaussian(coefficients[randomCoefficientIndex], sigma);
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

    static class NullBreeder implements Breeder
    {
        @Override
        public Individual breed(Individual first, Individual second)
        {
            return first;
        }
    }
}
