package org.example;

import java.util.Arrays;
import org.apache.commons.math3.distribution.GeometricDistribution;
import org.apache.commons.math3.random.RandomDataGenerator;

public class Main {
    static int POPULATION_SIZE = 1000;
    // found 8 with guess and check for a value which only rarely produces values near populationSize
    static double desiredExpectedValue = (double) POPULATION_SIZE / 100;
    static double p = 1/(desiredExpectedValue + 1);
    static GeometricDistribution geometricDistribution = new GeometricDistribution(p);
    static BoundedGeometricDistribution BOUNDED_GEOMETRIC_DIST = new BoundedGeometricDistribution(geometricDistribution);
    static RandomDataGenerator RANDOM = new RandomDataGenerator();

    public static void main(String[] args) {
        geneticAlgorithm();
    }

    static void geneticAlgorithm() {
        Population currentPopulation = Population.generateRandomPopulation(1000);

        for (int i = 0; i < 100; i++) {
            Individual[] nextGenerationIndividuals = new Individual[POPULATION_SIZE];

            for (int j = 0; j < POPULATION_SIZE; j++) {
                Individual sample1 = currentPopulation.sampleFitIndividual();
                Individual sample2 = currentPopulation.sampleFitIndividual();
                Individual child = sample1.breed(sample2);
                Individual mutatedChild = child.mutate();
                nextGenerationIndividuals[j] = mutatedChild;
            }

            // todo the algorithm is falling into a local minimum quite quickly. I tried to add mutations, but that only
            //  made things worse ... need to toy with the parameters. would be worth it to make a parameters object?

            currentPopulation = new Population(nextGenerationIndividuals);
            System.out.println(currentPopulation.averageFitness());
        }

        System.out.println(currentPopulation.individuals[0]);
    }

    record BoundedGeometricDistribution(GeometricDistribution geometricDistribution) {
        int sample(int upperBoundExclusive) {
            int sample = geometricDistribution.inverseCumulativeProbability(Math.random());
            if (sample >= upperBoundExclusive) {
                return sample(upperBoundExclusive);
            }

            return sample;
        }
    }

    static final double[] TRUE_COEFFICIENTS = new double[]{1, -1, -4, -9, 6.5, 9.5};

    static double evaluateFitness(double[] coefficients) {
        double sum = 0;
        for (double x = -2; x <= 2; x+= (double) 4 /20) {
            double trueValue = quinticPolynomial(x, TRUE_COEFFICIENTS);
            double individualsValue = quinticPolynomial(x, coefficients);
            sum += Math.pow(trueValue - individualsValue, 2);
        }

        return Math.pow(sum, 0.5);
    }

    /**
     * <a href="https://www.desmos.com/calculator/tuaumcnl3p">desmos</a>
     */
    static double quinticPolynomial(double x, double[] coefs) {
        return coefs[0] + coefs[1]*x + coefs[2]*Math.pow(x, 2) + coefs[3]*Math.pow(x, 3) + coefs[4]*Math.pow(x, 4) +
                coefs[5]*Math.pow(x, 5);
    }

    record Individual(double[] chromosome, double fitness) implements Comparable<Individual> {

        public Individual(double[] chromosome) {
            this(chromosome, evaluateFitness(chromosome));
        }

        public static Individual generateRandomIndividual() {
            double a = Math.random() * 20 - 10;
            double b = Math.random() * 20 - 10;
            double c = Math.random() * 20 - 10;
            double d = Math.random() * 20 - 10;
            double e = Math.random() * 20 - 10;
            double f = Math.random() * 20 - 10;
            double[] genome = new double[]{a, b, c, d, e, f};

            return new Individual(genome);
        }

        public Individual breed(Individual other) {
            double[] childCoefficients = new double[6];

            for (int i = 0; i < childCoefficients.length; i++) {
                if (Math.random() > 0.5) {
                    childCoefficients[i] = chromosome[i];
                } else {
                    childCoefficients[i] = other.chromosome[i];
                }
            }

            return new Individual(childCoefficients);
        }

        @Override
        public String toString() {
            return String.format("fitness: %f, chromosome: %s", fitness, Arrays.toString(chromosome));
        }

        @Override
        public int compareTo(Individual o) {
            double compare = fitness - o.fitness;
            if (compare < 0)
                return -1;
            else if (compare > 0)
                return 1;
            else
                return 0;
        }

        public Individual mutate() {
            double[] coefficients = new double[6];

            if (Math.random() > 0.3) {
                for (int i = 0; i < coefficients.length; i++) {
                    if (Math.random() > 0.3) {
                        coefficients[i] = RANDOM.nextGaussian(chromosome[i], 5);
                        coefficients[i] = Math.random() * 20 - 10;
                    }
                }
            }

            return new Individual(coefficients);
        }
    }

    record Population(Individual[] individuals)
    {
        public Population(Individual[] individuals) {
            Arrays.sort(individuals);
            this.individuals = individuals;
        }

        public Individual sampleFitIndividual(){
            int sampleIndex = BOUNDED_GEOMETRIC_DIST.sample(individuals.length);
            return individuals[sampleIndex];
        }

        public static Population generateRandomPopulation(int size){
            Individual[] individuals = new Individual[size];
            for (int i = 0; i < size; i++) {
                individuals[i] = Individual.generateRandomIndividual();
            }

            return new Population(individuals);
        }

        @Override
        public String toString() {
            StringBuilder builder = new StringBuilder();

            for (Individual individual : individuals) {
                builder.append(individual.toString());
                builder.append("\n");
            }

            return builder.toString();
        }

        public double averageFitness() {
            double sum = 0;
            for (Individual individual : individuals) {
                sum += individual.fitness;
            }
            return sum / individuals.length;
        }
    }
}
