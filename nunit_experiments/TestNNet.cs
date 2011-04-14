/*
 * Created by SharpDevelop.
 * User: tims
 * Date: 2011-04-13
 * Time: 2:24 PM
 * 
 * To change this template use Tools | Options | Coding | Edit Standard Headers.
 */
using System;
using NUnit.Framework;

using com.abznak.evolve;

namespace com.abznak.nnet
{

	public class EvolvingDouble :Evolveable<EvolvingDouble> {
		public double d {get; private set;}
		public EvolvingDouble(double d){
			this.d = d;
		}
		public double getFitness() {
			return d;			
		}
		public EvolvingDouble makeChild(MutationFunction mf) {
			return new EvolvingDouble(mf(d));
		}
	}
	[TestFixture]	
	public class TestEvolvingDouble {
		[Test]
		public static void Test() {
			EvolvingDouble ed1 = new EvolvingDouble(1);
			Assert.AreEqual(1, ed1.d);
			EvolvingDouble ed2 = ed1.makeChild((double d) => d + 1);
			Assert.AreEqual(2, ed2.d);		
		}

	}
	[TestFixture]
	public class TestFunctionFitter {
		
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
			Assert.AreEqual(1, hc.indiv.getFitness());
			Assert.AreEqual(0, hc.generation, "generation should start at 0");
			Assert.AreEqual(0, hc.better_count, "better_count should start at 0");
		}
		
		[Test]
		public void TestWorseTick() {
			hc.tick((double d) => d - 1);
			Assert.AreSame(ed, hc.indiv, "don't change the individual if new one is worse");
			Assert.AreEqual(1, hc.indiv.getFitness(), "fitness should not change if new indiv is worse");
			Assert.AreEqual(1, hc.generation, "generation should increase after tick");
			Assert.AreEqual(0, hc.better_count, "better_count should not increase with worse indiv");
		}

		[Test]
		public void TestBetterTick() {
			hc.tick((double d) => d + 1);
			Assert.AreNotSame(ed, hc.indiv, "do change the individual if new one is better");
			Assert.AreEqual(2, hc.indiv.getFitness(), "fitness should change if new indiv is better");
			Assert.AreEqual(1, hc.generation, "generation should increase after tick");			
			Assert.AreEqual(1, hc.better_count, "better_count should increase with better indiv");			
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
			double[][][] want_weights = makeWeights();
			double[][][] got_weights = FeedForwardNNet.cloneWeights(want_weights);
			Assert.AreSame(want_weights, want_weights);
			Assert.AreNotSame(want_weights, got_weights);
			Assert.AreEqual(want_weights, got_weights);
			got_weights[0][0][0] = 1000;
			Assert.AreNotEqual(want_weights, got_weights);
		
		}
		[Test]
		public void TestConstruct()
		{
			makeSample();
			Assert.AreEqual(want_in, nn.input_count);
			Assert.AreEqual(want_out, nn.output_count);
			//want_weights[1][1][1] = 15;
			Assert.AreEqual(want_weights, nn.weights);		
		}
		
		[Test]
		public void TestAvg() {
			makeAvgNN();
			double a = 6, b = 10;
			double[] output = nn.process(new double[] {a, b});
			Assert.AreEqual(want_out, output.Length);
			Assert.AreEqual((a + b)/2, output[0]);
		}

		public void makeSample()
		{
			want_in = 2;
			want_out = 3;
			want_weights = makeWeights();
			nn = new FeedForwardNNet(want_in, want_out, makeWeights(), NNet.AF_ID);
		}
		private double[][][] makeWeights() {
			return new double[][][] {
				new double[][] { new double[]{5,2,3}, new double[]{1,2,3}, new double[]{1,2,3} },
				new double[][] { new double[]{1,2,3}, new double[]{2,2,3}, new double[]{1,2,3} },
				new double[][] { new double[]{1,2,3}, new double[]{1,2,3}, new double[]{1,2,7} },
			};			
			
		}
		
		[Test]
		public void makeOffsetNN() {
			want_in = 1;
			want_out = 1;
			double offset = 13.5;
			want_weights = new double[][][] {
				new double[][] { new double[] {0, offset}}
			};
			nn = new FeedForwardNNet(want_in, want_out, want_weights, NNet.AF_ID);
			double[] got = nn.process(new double[] {5});
			Assert.AreEqual(offset, got[0]);
		}
		private void makeAvgNN() {
			want_in = 2;
			want_out = 1;
			want_weights = new double[][][] {
				new double[][] { new double[] {.5, .5, 0}}
			};
			nn = new FeedForwardNNet(want_in, want_out, want_weights, NNet.AF_ID);
		}
		[Test]
		public void makeMultiLevelNN() {
			want_in = 1;
			want_out = 1;
			want_weights = new double[][][] {
				new double[][] { new double[] {2, 0} },
				new double[][] { new double[] {3, 0} },				
			};
			nn = new FeedForwardNNet(want_in, want_out, want_weights, NNet.AF_ID);
			double[] got = nn.process(new double[] {5});		
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
