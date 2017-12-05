using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class SoftmaxLayer : layer
    {
        //int num_inputs;
        //int out_depth;
        //int out_sx;
        //int out_sy;
        //String layer_type = "softmax";
        //Vol in_act;
        //Vol out_act;
        double[] es;
        public SoftmaxLayer(Layer_def opt)
        {

            // computed
            this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
            out_depth = this.num_inputs;
            this.out_sx = 1;
            this.out_sy = 1;
            this.layer_type = "softmax";
            es = Convnet_util.zeros(out_depth);
        }

        public override Vol forward(Vol V, bool is_training)
        {
            this.in_act = V;

            var A = new Vol(1, 1, this.out_depth, 0.0);

            // compute max activation
            var _as = V.w;
            var amax = V.w[0];
            for (var i = 1; i < this.out_depth; i++)
            {
                if (_as[i] > amax) amax = _as[i];
            }

            // compute exponentials (carefully to not blow up)
            var es = Convnet_util.zeros(this.out_depth);
            var esum = 0.0;
            for (var i = 0; i < this.out_depth; i++)
            {
                var e = Math.Exp(_as[i] - amax);
                esum += e;
                es[i] = e;
            }

            // normalize and output to sum to one
            for (var i = 0; i < this.out_depth; i++)
            {
                es[i] /= esum;
                A.w[i] = es[i];
            }

            this.es = es; // save these for backprop
            this.out_act = A;
            return this.out_act;
        }
        public override double backward(int y)
        {

            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.in_act;
            x.dw = Convnet_util.zeros(x.w.Length); // zero out the gradient of input Vol

            for (var i = 0; i < this.out_depth; i++)
            {
                var indicator = i == y ? 1.0 : 0.0;
                var mul = -(indicator - this.es[i]);
                x.dw[i] = mul;
            }
            if (es.Length==0)
                return 0;
            double _es = es[y];
            // loss is the class negative log likelihood
            return -Math.Log(_es);
        }
        public override void backward() { }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }
    }
    class RegressionLayer : layer
    {
        // implements an L2 regression cost layer,
        // so penalizes \sum_i(||x_i - y_i||^2), where x is its input
        // and y is the user-provided array of "correct" values.
        int num_inputs;
        int out_depth;
        int out_sx;
        int out_sy;
        String layer_type = "regression";
        Vol in_act;
        Vol out_act;
        public RegressionLayer(Layer_def opt)
        {

            // computed
            this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
            this.out_depth = this.num_inputs;
            this.out_sx = 1;
            this.out_sy = 1;
            this.layer_type = "regression";
        }

        public override Vol forward(Vol V, bool is_training)
        {
            this.in_act = V;
            this.out_act = V;
            return V; // identity function
        }
        // y is a list here of size num_inputs
        // or it can be a number if only one value is regressed
        // or it can be a struct {dim: i, val: x} where we only want to 
        // regress on dimension i and asking it to have value x
        public override double backward(int y)
        {

            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.in_act;
            x.dw = Convnet_util.zeros(x.w.Length); // zero out the gradient of input Vol
            var loss = 0.0;
            /*

            if (y instanceof Array || y instanceof Float64Array) {
                for (var i = 0; i < this.out_depth; i++)
                {
                    var dy = x.w[i] - y[i];
                    x.dw[i] = dy;
                    loss += 0.5 * dy * dy;
                }
            } 
            else if (typeof y == "number")
            {
                // lets hope that only one number is being regressed
                var dy = x.w[0] - y;
                x.dw[0] = dy;
                loss += 0.5 * dy * dy;
            }
            else
            {
                // assume it is a struct with entries .dim and .val
                // and we pass gradient only along dimension dim to be equal to val
                var i = y.dim;
                var yi = y.val;
                var dy = x.w[i] - yi;
                x.dw[i] = dy;
                loss += 0.5 * dy * dy;
            }
            */
            return loss;
        }
        public override void backward() { }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }
    }
    class SVMLayer : layer
    {
        int num_inputs;
        int out_depth;
        int out_sx;
        int out_sy;
        String layer_type = "svm";
        Vol in_act;
        Vol out_act;
        public SVMLayer(Layer_def opt)
        {
            // computed
            this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
            this.out_depth = this.num_inputs;
            this.out_sx = 1;
            this.out_sy = 1;
            this.layer_type = "svm";
        }

        public override Vol forward(Vol V, bool is_training)
        {
            this.in_act = V;
            this.out_act = V; // nothing to do, output raw scores
            return V;
        }
        public override double backward(int y)
        {
            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.in_act;
            x.dw = Convnet_util.zeros(x.w.Length); // zero out the gradient of input Vol

            // we're using structured loss here, which means that the score
            // of the ground truth should be higher than the score of any other 
            // class, by a margin
            var yscore = x.w[y]; // score of ground truth
            var margin = 1.0;
            var loss = 0.0;
            for (var i = 0; i < this.out_depth; i++)
            {
                if (y == i) { continue; }
                var ydiff = -yscore + x.w[i] + margin;
                if (ydiff > 0)
                {
                    // violating dimension, apply loss
                    x.dw[i] += 1;
                    x.dw[y] -= 1;
                    loss += ydiff;
                }
            }

            return loss;
        }
        public override void backward() { }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }
    }

}
