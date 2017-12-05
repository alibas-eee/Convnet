using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
   
    class _array
    {
        public double[] data;
        public _array(int n)
        {
            data = new double[n];
        }
    }

    class Options
    {
        public double learning_rate;
        public double l1_decay;
        public double l2_decay;
        public double batch_size;
        public string method;

        public double momentum;
        public double ro;
        public double eps;
        public double beta1;
        public double beta2;
        public Options()
        {


        }
    }


    class Trainer
    {
        double learning_rate;
        double l1_decay;
        double l2_decay;
        double batch_size;
        string method;

        double momentum;
        double ro;
        double eps;
        double beta1;
        double beta2;
        bool regression;

        int k = 0; // iteration counter
        List<_array> gsum;  // last iteration gradients (used for momentum calculations)
        List<_array> xsum; // used in adam or adadelta

        Net net;
        public Trainer(Net net, String opt)
        {
            this.net = net;
            Options options = JsonConvert.DeserializeObject<Options>(opt);

            this.learning_rate = options.learning_rate != 0 ? options.learning_rate : 0.01;
            this.l1_decay = options.l1_decay != 0 ? options.l1_decay : 0.0;
            this.l2_decay = options.l2_decay != 0 ? options.l2_decay : 0.0;
            this.batch_size = options.batch_size != 0 ? options.batch_size : 1;
            this.method = options.method != null ? options.method : "sgd"; // sgd/adam/adagrad/adadelta/windowgrad/netsterov

            this.momentum = options.momentum != 0 ? options.momentum : 0.9;
            this.ro = options.ro != 0 ? options.ro : 0.95; // used in adadelta
            this.eps = options.eps != 0 ? options.eps : 1e-8; // used in adam or adadelta
            this.beta1 = options.beta1 != 0 ? options.beta1 : 0.9; // used in adam
            this.beta2 = options.beta2 != 0 ? options.beta2 : 0.999; // used in adam

            this.k = 0; // iteration counter
            this.gsum = new List<_array>(); // last iteration gradients (used for momentum calculations)
            this.xsum = new List<_array>(); // used in adam or adadelta

            // check if regression is expected 
            if (this.net.layers[this.net.layers.Count - 1].layer_type == "regression")
                this.regression = true;
            else
                this.regression = false;
        }

        public String train(Vol x, int y)
        {

            var start = DateTime.Now;
            this.net.forward(x, true); // also set the flag that lets the net know we're just training
            var end = DateTime.Now;
            var fwd_time = end - start;

            start = DateTime.Now;
            var cost_loss = this.net.backward(y);
            var l2_decay_loss = 0.0;
            var l1_decay_loss = 0.0;
            end = DateTime.Now;
            var bwd_time = end - start;

            // if (this.regression && y.constructor != Array)
            //    Console.WriteLine("Warning: a regression net requires an array as training output vector.");

            this.k++;
            if (this.k % this.batch_size == 0)
            {
                var pglist = net.getParamsAndGrads();                

                // initialize lists for accumulators. Will only be done once on first iteration
                if (this.gsum.Count == 0 && (this.method != "sgd" || this.momentum > 0.0))
                {
                    // only vanilla sgd doesnt need either lists
                    // momentum needs gsum
                    // adagrad needs gsum
                    // adam and adadelta needs gsum and xsum
                    for (var i = 0; i < pglist.Count; i++)
                    {

                        this.gsum.Add(new _array(pglist[i].param.Length));
                        if (this.method == "adam" || this.method == "adadelta")
                        {
                            this.xsum.Add(new _array(pglist[i].param.Length));
                        }
                        else
                        {
                            this.xsum.Add(new _array(1)); // conserve memory
                        }
                    }
                }

                // perform an update for all sets of weights
                for (var i = 0; i < pglist.Count; i++)
                {
                    var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
                    var p = pg.param;
                    var g = pg.grads;

                    // learning rate for some parameters.
                    var l2_decay_mul = pg.l2_decay_mul != 0 ? pg.l2_decay_mul : 1.0;
                    var l1_decay_mul = pg.l1_decay_mul != 0 ? pg.l1_decay_mul : 1.0;
                    var l2_decay = this.l2_decay * l2_decay_mul;
                    var l1_decay = this.l1_decay * l1_decay_mul;

                    var plen = p.Length;
                    for (var j = 0; j < plen; j++)
                    {
                        l2_decay_loss += l2_decay * p[j] * p[j] / 2; // accumulate weight decay loss
                        l1_decay_loss += l1_decay * Math.Abs(p[j]);
                        var l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
                        var l2grad = l2_decay * (p[j]);

                        var gij = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient

                        var gsumi = this.gsum[i].data;
                        var xsumi = this.xsum[i].data;
                        if (this.method == "adam")
                        {
                            // adam update
                            gsumi[j] = gsumi[j] * this.beta1 + (1 - this.beta1) * gij; // update biased first moment estimate
                            xsumi[j] = xsumi[j] * this.beta2 + (1 - this.beta2) * gij * gij; // update biased second moment estimate
                            var biasCorr1 = gsumi[j] * (1 - Math.Pow(this.beta1, this.k)); // correct bias first moment estimate
                            var biasCorr2 = xsumi[j] * (1 - Math.Pow(this.beta2, this.k)); // correct bias second moment estimate
                            var dx = -this.learning_rate * biasCorr1 / (Math.Sqrt(biasCorr2) + this.eps);
                            p[j] += dx;
                        }
                        else if (this.method == "adagrad")
                        {
                            // adagrad update
                            gsumi[j] = gsumi[j] + gij * gij;
                            var dx = -this.learning_rate / Math.Sqrt(gsumi[j] + this.eps) * gij;
                            p[j] += dx;
                        }
                        else if (this.method == "windowgrad")
                        {
                            // this is adagrad but with a moving window weighted average
                            // so the gradient is not accumulated over the entire history of the run. 
                            // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                            gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
                            var dx = -this.learning_rate / Math.Sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
                            p[j] += dx;
                        }
                        else if (this.method == "adadelta")
                        {
                            gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
                            var dx = -Math.Sqrt((xsumi[j] + this.eps) / (gsumi[j] + this.eps)) * gij;
                            xsumi[j] = this.ro * xsumi[j] + (1 - this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
                            p[j] += dx;
                        }
                        else if (this.method == "nesterov")
                        {
                            var dx = gsumi[j];
                            gsumi[j] = gsumi[j] * this.momentum + this.learning_rate * gij;
                            dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
                            p[j] += dx;
                        }
                        else
                        {
                            // assume SGD
                            if (this.momentum > 0.0)
                            {
                                // momentum update
                                var dx = this.momentum * gsumi[j] - this.learning_rate * gij; // step
                                gsumi[j] = dx; // back this up for next iteration of momentum
                                p[j] += dx; // apply corrected gradient
                            }
                            else
                            {
                                // vanilla sgd
                                p[j] += -this.learning_rate * gij;
                            }
                        }
                        g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
                    }
                }
            }

            // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
            // in future, TODO: have to completely redo the way loss is done around the network as currently 
            // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
            // and it should all be computed correctly and automatically. 
            String json = "{fwd_time:" + fwd_time +
                      ",bwd_time: " + bwd_time +
                      ",l2_decay_loss: " + l2_decay_loss +
                      ",l1_decay_loss: " + l1_decay_loss +
                      ",cost_loss: " + cost_loss +
                      ",softmax_loss: " + cost_loss +
                      ",loss: " + cost_loss + l1_decay_loss + l2_decay_loss + "}";
            return json;
        }
    }
}
