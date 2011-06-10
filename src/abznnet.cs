/*
 * abznnet - C# Neural Network Library
 * Copyright (C) 2011 Tim Smith <abznnet@abznak.com>
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */
 
/*
 * AbzNNet - Abznak's Neural Network Library.
 * This library is intended to be used when you want to quickly add
 * neural network and/or evolutionary capabilities to a project.  See README.md
 * for details.
 */
using System;
using System.IO;

namespace Abznak.Evolve {
	public class Util {
		public static readonly MutationFunction MF_SMALL_RND = (double d) => d + Util.rnd(-.05, .05);		
		private static Random random = new Random();
		public static double rnd(double min, double max) {
			return random.NextDouble() * (max - min) + min;		
		}
	}
	
	
	/// <summary>
	/// function for changing the innards of a copy of an evolveable object.
	/// TODO: it's a v0.x, I'll worry about mutating things with non-double innards if and when necessary.
	/// </summary>
	public delegate double MutationFunction(double input);		
	
	public interface IMutateable<T> {
		T makeChild(MutationFunction mf);		
	}
	public interface IEvolveable<T> : IMutateable<T> {
		/// <summary>
		/// get the fitness of the object. Higher fitness is better.
		/// </summary>
		/// <returns></returns>
		double getFitness();
	}
	public interface IEvolvableFactory<T> {
		T makeIndiv();
	}
	public class HillClimber<T> where T : IEvolveable<T> {
		
		
		public T indiv {get; private set;}
		public int generation {get; private set;}
		public int better_count {get; private set;}
		
					
		public HillClimber(T first) {
			this.indiv = first;						
			this.generation = 0;
			
		}
		/// <summary>
		/// run another generateion
		/// </summary>
		/// <param name="mf">mutation function to use on the indiv</param>
		/// <returns>new fitness</returns>
		public double tick(MutationFunction mf) {
			T newguy = indiv.makeChild(mf);
			double old_fitness = indiv.getFitness();
			double new_fitness = newguy.getFitness();
			double fitness = old_fitness;
			if (new_fitness > old_fitness) {
				indiv = newguy;
				fitness = new_fitness;
				better_count++;
			}
			generation++;
			return fitness;
		}				
	}

	public class Population<U,T> where U : IEvolveable<T> {
		
	}
	
	/*public interface CoEvolveable<T> : Evolveable<T> {		
	}*/
}

namespace Abznak.NeuralNet
{
	using Abznak.Evolve;
	
	/// <summary>
	/// activation function is used to scale the output of a neuron 
	/// </summary>
	public delegate double ActivationFunction(double input);
	public delegate double DoubleFunction(double input);	

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
	
	public class FunctionFitter : Evolve.IEvolveable<FunctionFitter> {
		
		public struct RangeSpec {
			public readonly double min;
			public readonly double max;
			public readonly double count;
			public RangeSpec(double min, double max, double count) {
				this.min = min;
				this.max = max;
				this.count = count;
			}
			public double getSample() {
				return Util.rnd(min, max);
				
			}
		}
		
		
		
		public FeedForwardNNet nnet {get; private set;}
		public DoubleFunction fn {get; private set;}
		/*public double range_min {get; private set;}
		public double range_max {get; private set;}
		public int range_count {get; private set;}*/
		public RangeSpec range {get; private set;}

		public FunctionFitter(FeedForwardNNet nnet, DoubleFunction fn, RangeSpec range) {
			this.nnet = nnet;
			this.fn = fn;
			this.range = range;
		}
		public FunctionFitter makeChild(Evolve.MutationFunction mf) {			
			return new FunctionFitter(nnet.makeChild(mf), fn, range);
		}
		
		static int generation = 0;
		/// <summary>
		/// returns negative of square of error between the NN and the fn
		/// </summary>
		/// <returns></returns>
		public double getFitness() {
			return getFitness("", null);
		}
		public double getFitness(string prefix, StreamWriter log) {			
			long debug_time = DateTime.Now.Ticks;
			
			double tot = 0;			
			for (int i = 0; i < range.count; i++) {
				double sample = range.getSample();
				double want = fn(sample);
				double got = nnet.process(new double[] { sample })[0];
				if (log != null) {
					log.WriteLine("{0}, {1}, {2}, {3}, {4}", prefix, debug_time, sample, want, got);
					
				}
				tot += Math.Pow(want - got,2);
			}
			return -tot/range.count;
		}			
		
	}
	public class FeedForwardNNet : NNet, Evolve.IMutateable<FeedForwardNNet> {
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
		/// <param name="weights">weights by layer, dst neuron, src neuron. at the end of each list of weights is an offset (i.e. a weight for a neuron that always outputs 1).</param>
		public FeedForwardNNet(double[][][] weights, ActivationFunction af) {
			this.input_count = weights[0][0].Length - 1;  //-1 to account for offset weight
			this.output_count = weights[weights.Length - 1].Length;
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
		public FeedForwardNNet makeChild(Evolve.MutationFunction mf) {
			return new FeedForwardNNet(mutateWeights(weights, mf), af);
		}
		
		/// <summary>
		/// make a deep clone of a weights array
		/// </summary>
		/// <param name="weights">a ragged array of weights</param>
		/// <returns>a copy of weights, with no shared references</returns>
		public static double[][][] cloneWeights(double[][][] weights) {
			return mutateWeights(weights, (double d) => d); //less efficient that possible, but I'd rather that than duplicate code
		}
		public static double[][][] mutateWeights(double[][][] weights, MutationFunction fn) {			
			double[][][] ret = new double[weights.Length][][];
			for (int li = 0; li < weights.Length; li++) {			
				double[][] old_layer = weights[li];
				double[][] new_layer = new double[old_layer.Length][];
				for (int di = 0; di < old_layer.Length; di++) {
					//can clone the last bit, because there are no references in a double[]
					//new_layer[di] = (double[])old_layer[di].Clone();
					double [] old_unit = old_layer[di];
					double [] new_unit = new double[old_unit.Length];
					for (int si = 0; si < old_unit.Length; si++) {
						new_unit[si] = fn(old_unit[si]);
					}
					new_layer[di] = new_unit;
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
