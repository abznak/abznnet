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
	public interface Evolveable<T> {
		T makeChild(double mutationRate);
	}	
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
	
	public class FeedForwardNNet : NNet, evolve.Evolveable<FeedForwardNNet> {
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
		
		public FeedForwardNNet makeChild(double mutationRate) {
			return null;
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
