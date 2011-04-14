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

namespace com.abznak.nnet
{
	
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
