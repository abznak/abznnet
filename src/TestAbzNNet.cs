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
 * Tests for AbzNNet.
 * 
 * Note: Tests both Abznak.NeuralNet and Abznak.Evolve under the Abznak.NeuralNet
 * namespace because I wanted to stick to a single file and it doesn't work
 * with two namespaces in one file.
 */
using System;
using NUnit.Framework;
using System.IO;

using Abznak.Evolve;

namespace Abznak.NeuralNet
{

	public class EvolvingDouble :IEvolveable<EvolvingDouble> {
		public double d {get; private set;}
		public EvolvingDouble(double d){
			this.d = d;
		}
		public double GetFitness() {
			return d;			
		}
		public EvolvingDouble MakeChild(MutationFunction mf) {
			return new EvolvingDouble(mf(d));
		}
	}
	[TestFixture]	
	public class TestEvolvingDouble {
		[Test]
		public static void Test() {
			EvolvingDouble ed1 = new EvolvingDouble(1);
			Assert.AreEqual(1, ed1.d);
			EvolvingDouble ed2 = ed1.MakeChild((double d) => d + 1);
			Assert.AreEqual(2, ed2.d);		
		}

	}
	[TestFixture]
	public class TestFunctionFitter {
		private FunctionFitter.RangeSpec range;
		private FeedForwardNNet nn;
		public FunctionFitter ff;
		double range_min, range_max, range_cnt;

		[Test]
		public void MakeFFRange() {
			range_min = -2*Math.PI;
			range_max = -range_min;
			range_cnt = 1000;
			
			range = new FunctionFitter.RangeSpec(range_min, range_max, range_cnt);
			Assert.AreEqual(range_min, range.min, "diff: " + (range_min - range.min));
			Assert.AreEqual(range_max, range.max);
			Assert.AreEqual(range_cnt, range.count);
			
		}
		[Test]
		public void TestRange() {
			MakeFFRange();
			for (int i = 0; i < 1000; i++) {
				double sample = range.GetSample();
				Assert.That(sample, Is.GreaterThanOrEqualTo(range_min));
				Assert.That(sample, Is.LessThanOrEqualTo(range_max));				
			}
			
		}
		
		
		
		[Test]
		public void TestFF() {
			//makeFF();
			
		}
		
		[Test]
		public void MakeFFImperfect() {
			MakeFF((double d) => d * 1 + 3);
			double fitness = ff.GetFitness();
			Assert.That(fitness, Is.LessThan(0));
		}
		
		[Test]
		public void MakeFFPerfect() {
			MakeFF((double d) => d * 2 + .5);
			double fitness = ff.GetFitness();
			Assert.AreEqual(0, fitness);
				
		}
		public void MakeFF(DoubleFunction fn) {				
			MakeFFRange();			
			var weights = new double[][][] {
				new double[][] { new double[] { 2, .5} }  // d * 2 + .5
			};			
			nn = new FeedForwardNNet(weights, NNet.AF_ID);			
			ff = new FunctionFitter(nn, fn, range);
			Assert.AreSame(nn, ff.nnet);
			Assert.AreSame(fn, ff.fn);									
			//nn = new FeedForwardNNet(			
			
		}
		
		[Test]
		public void TestFit() {
			//ff.
		}
		
		
	}
	
	[TestFixture]
	public class TestHillClimber {
		private EvolvingDouble ed;
		private HillClimber<EvolvingDouble> hc;
		
		[Test]
		[TestFixtureSetUp]
		public void TestCtor() {
			ed = new EvolvingDouble(1);
			hc = new HillClimber<EvolvingDouble>(ed);
			Assert.AreSame(ed, hc.indiv);
			Assert.AreEqual(1, hc.indiv.GetFitness());
			Assert.AreEqual(0, hc.generation, "generation should start at 0");
			Assert.AreEqual(0, hc.betterCount, "betterCount should start at 0");
		}
		
		[Test]
		public void TestWorseTick() {
			hc.Tick((double d) => d - 1);
			Assert.AreSame(ed, hc.indiv, "don't change the individual if new one is worse");
			Assert.AreEqual(1, hc.indiv.GetFitness(), "fitness should not change if new indiv is worse");
			Assert.AreEqual(1, hc.generation, "generation should increase after tick");
			Assert.AreEqual(0, hc.betterCount, "betterCount should not increase with worse indiv");
		}

		[Test]
		public void TestBetterTick() {
			hc.Tick((double d) => d + 1);
			Assert.AreNotSame(ed, hc.indiv, "do change the individual if new one is better");
			Assert.AreEqual(2, hc.indiv.GetFitness(), "fitness should change if new indiv is better");
			Assert.AreEqual(1, hc.generation, "generation should increase after tick");			
			Assert.AreEqual(1, hc.betterCount, "betterCount should increase with better indiv");			
		}
		// TODO: makeChild funcions aren't tested, logging is hacked together, some things disabled for test
		// note to self - don't use stochastic /anything/ in tests unless absolutely necessary
		[Test]
		public void TestEvolvingFnFitter() {
			StreamWriter log = null;
			try {
				var tff = (new TestFunctionFitter());
				tff.MakeFFImperfect();
				FunctionFitter ff = tff.ff;
				var hc = new HillClimber<FunctionFitter>(ff);
				double oldf = hc.indiv.GetFitness();
				Console.WriteLine("START TestEvolvingFnFitter");
				long testcode = DateTime.Now.Ticks;
				log = new StreamWriter("c:\\tims\\tmp\\ntest2.csv", false);
				log.WriteLine("generation, time, sample, got, want");
				for (int i = 0; i < 10; i++) {
					hc.Tick(Util.MF_SMALL_RND);
					double f = hc.indiv.GetFitness(""+i, log);
					
//fitness is changing because getFitness is stochasitc
					
					//Assert.That(f, Is.GreaterThanOrEqualTo(oldf), "checking that fitness increased in generation " + i);
				}
				
			} finally {
				log.Close();
				Console.WriteLine("END TestEvolvingFnFitter");
				
			}
		}
		
		
		//[Test]
		public void MultiTestEvolvingFnFitter() {
			//TestEvolvingFnFitter has a 50% chance of getting a no-change on the first round, want a good chance of testing both paths
			for (int i = 0; i < 10; i++) {
				TestEvolvingFnFitter();
			}
		}
		
		
	}
	

		
		
		
	[TestFixture]
	public class TestNNet
	{
		[Test]
		public void TestAF_ID() {
			double[] ds = new double[] {0, 1, 100, -100, double.MaxValue, double.MinValue};
			foreach (double d in ds) {
				Assert.AreEqual(d, NNet.AF_ID(d));
			}
		}
		[Test]
		public void TestAF_TANH() {
			double[] ds = new double[] {0, 1, 100, -100, double.MaxValue, double.MinValue};
			foreach (double d in ds) {
				double got = NNet.AF_TANH(d);
				Assert.AreEqual(Math.Tanh(d), got);
				Assert.LessOrEqual(-1, got);
				Assert.GreaterOrEqual(1, got);				
			}
		}		
		
	}
	[TestFixture]
	public class TestFeedForwardNNet
	{

		private FeedForwardNNet nn;
		private int want_in, want_out;
		private double[][][] want_weights;
		

		[Test]
		public void TestCloneWeights() {
			double[][][] want_weights = MakeWeights();
			double[][][] got_weights = FeedForwardNNet.CloneWeights(want_weights);
			Assert.AreSame(want_weights, want_weights);
			Assert.AreNotSame(want_weights, got_weights);
			Assert.AreEqual(want_weights, got_weights);
			got_weights[0][0][0] = 1000;
			Assert.AreNotEqual(want_weights, got_weights);
		
		}
		[Test]
		public void TestConstruct()
		{
			MakeSample();
			Assert.AreEqual(want_in, nn.inputCount);
			Assert.AreEqual(want_out, nn.outputCount);
			//want_weights[1][1][1] = 15;
			Assert.AreEqual(want_weights, nn.weights);		
		}
		
		[Test]
		public void TestAvg() {
			MakeAvgNN();
			double a = 6, b = 10;
			double[] output = nn.Process(new double[] {a, b});
			Assert.AreEqual(want_out, output.Length);
			Assert.AreEqual((a + b)/2, output[0]);
		}

		public void MakeSample()
		{
			want_in = 2;
			want_out = 3;
			want_weights = MakeWeights();
			nn = new FeedForwardNNet(MakeWeights(), NNet.AF_ID);
		}
		/// <summary>
		/// make a set of weights with no particular effect
		/// </summary>
		/// <returns>3x3x3 array of doubles</returns>
		private double[][][] MakeWeights() {
			return new double[][][] {
				new double[][] { new double[]{5,2,3}, new double[]{1,2,3}, new double[]{1,2,3} },
				new double[][] { new double[]{1,2,3}, new double[]{2,2,3}, new double[]{1,2,3} },
				new double[][] { new double[]{1,2,3}, new double[]{1,2,3}, new double[]{1,2,7} },
			};			
			
		}
		
  		[Test]
		public void MakeOffsetNN() {
			want_in = 1;
			want_out = 1;
			double offset = 13.5;
			want_weights = new double[][][] {
				new double[][] { new double[] {0, offset}}
			};
			nn = new FeedForwardNNet( want_weights, NNet.AF_ID);
			double[] got = nn.Process(new double[] {5});
			Assert.AreEqual(offset, got[0]);
		}
		private void MakeAvgNN() {
			want_in = 2;
			want_out = 1;
			want_weights = new double[][][] {
				new double[][] { new double[] {.5, .5, 0}}
			};
			nn = new FeedForwardNNet(want_weights, NNet.AF_ID);
		}
		[Test]
		public void MakeMultiLevelNN() {
			want_in = 1;
			want_out = 1;
			want_weights = new double[][][] {
				new double[][] { new double[] {2, 0} },
				new double[][] { new double[] {3, 0} },				
			};
			nn = new FeedForwardNNet(want_weights, NNet.AF_ID);
			double[] got = nn.Process(new double[] {5});		
			Assert.AreEqual(2*3*5, got[0]);
		}
		
//		[TestFixture]
//		public void Dispose()
//		{
//			
//		}
//				
	}
}
