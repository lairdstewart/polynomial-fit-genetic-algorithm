package org.example;

import org.apache.commons.math3.distribution.GeometricDistribution;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

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
    // todo curious to try and see what optimal setup there is when we only have 1000 sample size
    // range -10 to 10
    private static final double[] TRUE_COEFFICIENTS = new double[]{1, -1, -4, -9, 6.5, 9.5};
    private static final RandomDataGenerator RANDOM = new RandomDataGenerator();
    private static final Parameters GEOMETRIC_POINTWISE = new Parameters(10000, 50, new GeometricSelector(0.01, 0.1), new PointwiseRandomMutator(), new PointwiseBreeder(), new RSSFitnessEvaluator()); // 80
    private static final Parameters TOP_GAUSSIAN_MID_SIGMA = new Parameters(10000, 50, new TopOfClassSelector(0.1), new PointwiseGaussianMutator(1), new PointwiseBreeder(), new RSSFitnessEvaluator()); // 8
    private static final Parameters TOP_GAUSSIAN_SMALL_SIGMA = new Parameters(1000, 50, new TopOfClassSelector(0.1),
            new PointwiseGaussianMutator(0.1), new PointwiseBreeder(), new RSSFitnessEvaluator()); // 1
    private static final Parameters TOP_NO_MUTATION = new Parameters(10000, 50, new TopOfClassSelector(0.1), new NullMutator(), new PointwiseBreeder(), new RSSFitnessEvaluator()); // 1
    private static final Parameters DO_NOTHING = new Parameters(10000, 50, new TopOfClassSelector(0.1), new NullMutator(), new NullBreeder(), new RSSFitnessEvaluator());
    private static int NUM_FITNESS_EVALUATIONS = 0;

    public static void main(String[] args)
    {
        List<double[]> chromosomeHistory = geneticAlgorithm(TOP_NO_MUTATION);
        saveToDat(chromosomeHistory);
        System.out.println("total polynomial evaluations: " + NUM_FITNESS_EVALUATIONS);
    }

    private static void saveToDat(List<double[]> arrays)
    {
        String fileName = "chromosome-history.dat";

        try (FileWriter fileWriter = new FileWriter(fileName); BufferedWriter bufferedWriter = new BufferedWriter(fileWriter))
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

    private static List<double[]> geneticAlgorithm(Parameters params)
    {
        List<double[]> chromosomeHistory = new ArrayList<>();

        double[][] currentPopulation = generateRandomPopulation(params.populationSize);
        double previousAverageFitness = Double.MAX_VALUE;

        for (int iteration = 0; iteration < params.numIterations; iteration++)
        {
            double[] fitness = calculateFitness(params, currentPopulation);
            double[][] sortedPopulation = sortPopulationByFitness(params, fitness, currentPopulation);

            // bookkeeping
            chromosomeHistory.add(sortedPopulation[0]);
            double averageFitness = Arrays.stream(fitness).average().orElse(0.0);
            System.out.println(averageFitness);

            // select parents todo: to make this more clear, select() should select parent pairs
            int[] selectedParents = params.selector().select(sortedPopulation);

            // create one child per parent pair via crossover and mutatation
            double[][] nextGeneration = new double[params.populationSize][6];

            for (int j = 0; j < params.populationSize; j++)
            {
                int firstIndex = RANDOM.nextInt(0, selectedParents.length - 1);
                int secondIndex;
                do
                {
                    secondIndex = RANDOM.nextInt(0, selectedParents.length - 1);
                } while (secondIndex == firstIndex);

                double[] selectedIndividual1 = sortedPopulation[firstIndex];
                double[] selectedIndividual2 = sortedPopulation[secondIndex];

                double[] childChromosome = params.breeder.breed(selectedIndividual1, selectedIndividual2);
                double[] mutatedChild = params.mutator().mutate(childChromosome);
                nextGeneration[j] = mutatedChild;
            }

            if (averageFitness == previousAverageFitness)
            {
                break;
            }

            currentPopulation = nextGeneration;
            previousAverageFitness = averageFitness;
        }

        return chromosomeHistory;
    }

    private static double[][] sortPopulationByFitness(Parameters params, double[] fitness, double[][] currentGeneration)
    {
        Integer[] indicesSortedByFitness = IntStream.range(0, params.populationSize).boxed().toArray(Integer[]::new);
        Arrays.sort(indicesSortedByFitness, Comparator.comparingDouble(o -> fitness[o]));
        double[][] sortedPopulation = new double[params.populationSize][6];
        for (int i = 0; i < currentGeneration.length; i++)
        {
            sortedPopulation[i] = currentGeneration[indicesSortedByFitness[i]];
        }
        return sortedPopulation;
    }

    private static double[] calculateFitness(Parameters params, double[][] currentGeneration)
    {
        double[] fitness = new double[params.populationSize];
        for (int i = 0; i < params.populationSize; i++)
        {
            fitness[i] = params.fitnessEvaluator.evaluate(currentGeneration[i]);
        }
        return fitness;
    }

    static double evaluateFitness(double[] coefficients)
    {
        NUM_FITNESS_EVALUATIONS++;

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

    private static double[] generateRandomIndividual()
    {
        double a = Math.random() * 20 - 10;
        double b = Math.random() * 20 - 10;
        double c = Math.random() * 20 - 10;
        double d = Math.random() * 20 - 10;
        double e = Math.random() * 20 - 10;
        double f = Math.random() * 20 - 10;
        return new double[]{a, b, c, d, e, f};
    }

    private static double[][] generateRandomPopulation(int size)
    {
        double[][] individuals = new double[size][6];
        for (int i = 0; i < size; i++)
        {
            individuals[i] = generateRandomIndividual();
        }

        return individuals;
    }

    interface FitnessEvaluator
    {
        double evaluate(double[] genome);
    }

    interface Breeder
    {
        double[] breed(double[] first, double[] second);
    }

    interface Mutator
    {
        double[] mutate(double[] individual);
    }

    interface Selector
    {
        /**
         * Assumes the population is sorted. Return the index of selected individuals.
         */
        int[] select(double[][] population);
    }

    static class SinglePointCrossover implements Breeder
    {
        @Override
        public double[] breed(double[] first, double[] second)
        {
            double[] childCoefficients = new double[6];

            int crossoverIndex = RANDOM.nextInt(0, 5);

            for (int i = 0; i < 6; i++)
            {
                if (i < crossoverIndex)
                {
                    childCoefficients[i] = first[i];
                } else
                {
                    childCoefficients[i] = second[i];
                }
            }

            return childCoefficients;
        }
    }

    static class PointwiseBreeder implements Breeder
    {
        @Override
        public double[] breed(double[] first, double[] second)
        {
            double[] childCoefficients = new double[6];

            for (int i = 0; i < childCoefficients.length; i++)
            {
                if (RANDOM.nextUniform(0, 1) > 0.5)
                {
                    childCoefficients[i] = first[i];
                } else
                {
                    childCoefficients[i] = second[i];
                }
            }

            return childCoefficients;
        }
    }

    /**
     * Residual sum of squares
     */
    static class RSSFitnessEvaluator implements FitnessEvaluator
    {
        @Override
        public double evaluate(double[] genome)
        {
            NUM_FITNESS_EVALUATIONS++;

            double sum = 0;
            for (double x = -2; x <= 2; x += (double) 4 / 20)
            {
                double trueValue = quinticPolynomial(x, TRUE_COEFFICIENTS);
                double individualsValue = quinticPolynomial(x, genome);
                sum += Math.pow(trueValue - individualsValue, 2);
            }

            return sum;
        }
    }

    private record Parameters(int populationSize, int numIterations, Selector selector, Mutator mutator,
                              Breeder breeder, FitnessEvaluator fitnessEvaluator)
    {
    }

    static class WeightedSelector implements Selector
    {
        @Override
        public int[] select(double[][] population)
        {
            return new int[0];
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
        public int[] select(double[][] population)
        {
            int numToSelect = (int) (population.length * pctToSelect);
            return IntStream.range(0, numToSelect).toArray();
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
        public int[] select(double[][] population)
        {
            int numToSelect = (int) (population.length * pctToSelect);

            boolean[] selectedIndices = new boolean[population.length];

            int numSelected = 0;
            while (numSelected < numToSelect)
            {
                int sampleIndex = sample(population.length);

                while (sampleIndex < population.length)
                {
                    if (!selectedIndices[sampleIndex])
                    {
                        selectedIndices[sampleIndex] = true;
                        numSelected++;
                        break;
                    }

                    // if already selected, select the next individual and so on
                    sampleIndex++;
                }
            }

            int[] result = new int[numSelected];
            int resultSize = 0;

            for (int i = 0; i < selectedIndices.length; i++)
            {
                if (selectedIndices[i])
                {
                    result[resultSize++] = i;
                }
            }

            return result;
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

    static class NullMutator implements Mutator
    {
        @Override
        public double[] mutate(double[] chromosome)
        {
            return chromosome;
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
        public double[] mutate(double[] chromosome)
        {
            if (RANDOM.nextUniform(0, 1) < 0.5)
            {
                double[] mutantChromosome = new double[chromosome.length];
                int randomCoefficientIndex = RANDOM.nextInt(0, 5);
                mutantChromosome[randomCoefficientIndex] = RANDOM.nextGaussian(chromosome[randomCoefficientIndex], sigma);
                return mutantChromosome;
            }

            return chromosome;
        }
    }

    static class PointwiseRandomMutator implements Mutator
    {
        @Override
        public double[] mutate(double[] chromosome)
        {
            if (RANDOM.nextUniform(0, 1) < 0.5)
            {
                double[] mutantChromosome = new double[chromosome.length];
                int randomCoefficientIndex = RANDOM.nextInt(0, 5);
                mutantChromosome[randomCoefficientIndex] = RANDOM.nextUniform(-10, 10);
                return mutantChromosome;
            }

            return chromosome;
        }
    }

    static class NullBreeder implements Breeder
    {
        @Override
        public double[] breed(double[] first, double[] second)
        {
            return first;
        }
    }
}
