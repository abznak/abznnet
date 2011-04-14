/*
 * Created by SharpDevelop.
 * User: tims
 * Date: 2011-04-13
 * Time: 2:06 PM
 * 
 * To change this template use Tools | Options | Coding | Edit Standard Headers.
 */
using System;

namespace com.abznak.evolve {
	/// <summary>
	/// function for changing the innards of a copy of an evolveable object.
	/// TODO: it's a v0.x, I'll worry about mutating things with non-double innards if and when necessary.
	/// </summary>
	public delegate double MutationFunction(double input);		
	
	public interface Mutateable<T> {
		T makeChild(MutationFunction mf);		
	}
	public interface Evolveable<T> : Mutateable<T> {
		/// <summary>
		/// get the fitness of the object. Higher fitness is better.
		/// </summary>
		/// <returns></returns>
		double getFitness();
	}
	public interface Factory<T> {
		T makeIndiv();
	}
	public class HillClimber<T> where T : Evolveable<T> {
		public T indiv {get; private set;}
		public int generation {get; private set;}
					
		public HillClimber(T first) {
			this.indiv = first;						
			this.generation = 0;
		}
		public void tick(MutationFunction mf) {
			T newguy = indiv.makeChild(mf);
			if (newguy.getFitness() > indiv.getFitness()) {
				indiv = newguy;
			}
		}				
	}

	public class Population<U,T> where U : Evolveable<T> {
		
	}
	
	/*public interface CoEvolveable<T> : Evolveable<T> {		
	}*/
}

namespace com.abznak.nnet
{
	/// <summary>
	/// activation function is used to scale the output of a neuron 
	/// </summary>
	public delegate double ActivationFunction(double input);

	/// <summary>
	/// Description of Class1.
	/// </summary>
	public abstract class NNet
	{
		/// <summary>
		/// the identity activation function
		/// </summary>
		public static readonly ActivationFunction AF_ID = delegate(double d) {return d;};  
		/// <summary>
		/// tanh is a nice activation function, it rescales output to be in [-1,1]
		/// </summary>
		public static readonly ActivationFunction AF_TANH = delegate(double d) {return Math.Tanh(d);};
		public abstract double[] process(double[] src);
		private static double id(double d) { return d;}
	}
	
	public class FeedForwardNNet : NNet, evolve.Mutateable<FeedForwardNNet> {
		public int input_count {get; private set;}
		public int output_count {get; private set;}
		public double[][][] weights {get; private set;}
		//public double[][] sums {get; private set;}
		public double[][] activations {get; private set;}
		public ActivationFunction af {get; private set;}

		/// <summary>
		/// Create feed forward, fully connected simulated Neural Network.
		/// For efficiency and lazieness reasons, weights isn't deep copied as much as it should be. so... don't mess with it.
		/// </summary>
		/// <param name="input_count">number of inputs</param>
		/// <param name="output_count">number of outputs</param>
		/// <param name="weights">weights by layer, dst neuron, src neuron. at the end of each list of weights is an offset (i.e. a weight for a neuron that always outputs 1).</param>
		public FeedForwardNNet(int input_count, int output_count, double[][][] weights, ActivationFunction af) {
			this.input_count = input_count;
			this.output_count = output_count;
			this.weights = weights;
			this.af = af;
		}
		/// <summary>
		/// checks that the weights array is sane (i.e. the sizes of the rows is correct)
		/// </summary>
		/// <returns>true iff weights array is OK</returns>
		public bool isSane() {
			throw new Exception("NYI");
		}
		public FeedForwardNNet makeChild(evolve.MutationFunction mf) {
			return null;
		}
		public FeedForwardNNet makeChild(double mutationRate) {
			return null;
		}
		
		/// <summary>
		/// make a deep clone of a weights array
		/// </summary>
		/// <param name="weights">a ragged array of weights</param>
		/// <returns>a copy of weights, with no shared references</returns>
		public static double[][][] cloneWeights(double[][][] weights) {
			double[][][] ret = new double[weights.Length][][];
			for (int li = 0; li < weights.Length; li++) {			
				double[][] old_layer = weights[li];
				double[][] new_layer = new double[old_layer.Length][];
				for (int di = 0; di < old_layer.Length; di++) {
					//can clone the last bit, because there are no references in a double[]
					new_layer[di] = (double[])old_layer[di].Clone();
				}
				ret[li] = new_layer;
			}
			return ret;
		}
		public override double[] process(double[] src) {
			activations = new double[weights.Length+1][];  //note that activations[0] is src, whereas weights[0] is hidden layer 1
			activations[0] = src; //TODO: deep clone
			
			//li - layer index
			//di - dst unit index
			//si - src unit index
			for (int li = 0; li < weights.Length; li++) {
				activations[li+1] = new double[weights[li].Length];
				double [][] layer = weights[li];
				for (int di = 0; di < layer.Length; di++) {
					double [] unit = layer[di];
					double sum = 0;
					int si;
					for (si = 0; si < unit.Length - 1; si++) {
						sum += unit[si] * activations[li][si];
					}
					sum += 1 * unit[si];  //offset
					activations[li+1][di] = af(sum);
				}
			}
			return activations[weights.Length];
		}
	}
}
