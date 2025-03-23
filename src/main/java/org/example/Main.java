package org.example;

import org.apache.commons.math3.distribution.GeometricDistribution;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
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
    private static final Parameters TOP_GAUSSIAN_SMALL_SIGMA = new Parameters(1000, 50, new TopOfClassSelector(0.1), new PointwiseGaussianMutator(0.1), new PointwiseBreeder(), new RSSFitnessEvaluator()); // 1
    private static final Parameters TOP_NO_MUTATION = new Parameters(10000, 50, new TopOfClassSelector(0.1), new NullMutator(), new PointwiseBreeder(), new RSSFitnessEvaluator()); // 1
    private static final Parameters DO_NOTHING = new Parameters(10000, 50, new TopOfClassSelector(0.1), new NullMutator(), new NullBreeder(), new RSSFitnessEvaluator());
    private static int NUM_FITNESS_EVALUATIONS = 0;

    public static void main(String[] args)
    {
        double[][] chromosomeHistory = geneticAlgorithm(TOP_NO_MUTATION);
        saveToDat(chromosomeHistory);
        System.out.println("total polynomial evaluations: " + NUM_FITNESS_EVALUATIONS);
    }

    private static void saveToDat(double[][] chromosomeHistory)
    {
        String fileName = "chromosome-history.dat";

        try (FileWriter fileWriter = new FileWriter(fileName); BufferedWriter bufferedWriter = new BufferedWriter(fileWriter))
        {
            for (double[] chromosome : chromosomeHistory)
            {
                for (int i = 0; i < chromosome.length - 1; i++)
                {
                    bufferedWriter.write(chromosome[i] + ",");
                }

                bufferedWriter.write(chromosome[chromosome.length - 1] + "");
                bufferedWriter.newLine();
            }
        } catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    private static double[][] geneticAlgorithm(Parameters params)
    {
        double[][] chromosomeHistory = new double[params.maxNumIterations][6];
        double[][] currentPopulation = generateRandomPopulation(params.populationSize);
        double previousAverageFitness = Double.MAX_VALUE;

        for (int iteration = 0; iteration < params.maxNumIterations; iteration++)
        {
            // core algorithm
            double[] fitness = calculateFitness(params, currentPopulation);
            double[][] sortedPopulation = sortPopulationByFitness(params, fitness, currentPopulation);
            double[][][] selectedParentPairs = params.selector.select(sortedPopulation);
            double[][] nextGeneration = breedNextGeneration(params, selectedParentPairs);

            // bookkeeping
            chromosomeHistory[iteration] = sortedPopulation[0];
            double averageFitness = Arrays.stream(fitness).average().orElse(0.0);
            System.out.println(averageFitness);

            // update for next gen
            if (averageFitness == previousAverageFitness)
            {
                break;
            }
            currentPopulation = nextGeneration;
            previousAverageFitness = averageFitness;
        }

        return chromosomeHistory;
    }

    private static double[][] breedNextGeneration(Parameters params, double[][][] selectedParentPairs)
    {
        double[][] nextGeneration = new double[params.populationSize][6];

        for (int i = 0; i < params.populationSize; i++)
        {
            double[] selectedParent1 = selectedParentPairs[i][0];
            double[] selectedParent2 = selectedParentPairs[i][1];
            double[] childChromosome = params.breeder.breed(selectedParent1, selectedParent2);
            double[] mutatedChild = params.mutator.mutate(childChromosome);
            nextGeneration[i] = mutatedChild;
        }
        return nextGeneration;
    }

    /**
     * @return double[numPairsToSample][parent index][gene index]
     */
    private static double[][][] sampleParentPairsWithoutReplacement(double[][] individuals, int numPairsToSample)
    {
        double[][][] result = new double[numPairsToSample][2][6];

        for (int i = 0; i < numPairsToSample; i++)
        {
            int firstIndex = RANDOM.nextInt(0, individuals.length - 1);
            int secondIndex;
            do
            {
                secondIndex = RANDOM.nextInt(0, individuals.length - 1);
            } while (secondIndex == firstIndex);

            result[i][0] = individuals[firstIndex];
            result[i][1] = individuals[secondIndex];
        }

        return result;
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
         * Assumes the population is sorted by fitness. Return pairs of parents to be bred.
         * return: double[population.length][num parents (2)][num genes (6)]
         */
        double[][][] select(double[][] population);
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

    private record Parameters(int populationSize, int maxNumIterations, Selector selector, Mutator mutator,
                              Breeder breeder, FitnessEvaluator fitnessEvaluator)
    {
    }

    static class TopOfClassSelector implements Selector
    {
        final double pctToSelect;

        TopOfClassSelector(double pctToSelect)
        {
            this.pctToSelect = pctToSelect;
        }

        @Override
        public double[][][] select(double[][] population)
        {
            int numParents = (int) (population.length * pctToSelect);
            double[][] parents = new double[numParents][6];
            System.arraycopy(population, 0, parents, 0, numParents);

            return sampleParentPairsWithoutReplacement(parents, population.length);
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
        public double[][][] select(double[][] population)
        {
            return null;

            // todo
//            int numToSelect = (int) (population.length * pctToSelect);
//
//            boolean[] selectedIndices = new boolean[population.length];
//
//            int numSelected = 0;
//            while (numSelected < numToSelect)
//            {
//                int sampleIndex = sample(population.length);
//
//                while (sampleIndex < population.length)
//                {
//                    if (!selectedIndices[sampleIndex])
//                    {
//                        selectedIndices[sampleIndex] = true;
//                        numSelected++;
//                        break;
//                    }
//
//                    // if already selected, select the next individual and so on
//                    sampleIndex++;
//                }
//            }
//
//            int[] result = new int[numSelected];
//            int resultSize = 0;
//
//            for (int i = 0; i < selectedIndices.length; i++)
//            {
//                if (selectedIndices[i])
//                {
//                    result[resultSize++] = i;
//                }
//            }
//
//            return result;
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
