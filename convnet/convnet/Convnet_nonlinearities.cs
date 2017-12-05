using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class ReluLayer:layer
    {
        public ReluLayer(Layer_def opt)
        {
            this.out_sx = opt.in_sx;
            this.out_sy = opt.in_sy;
            this.out_depth = opt.in_depth;
            this.layer_type = "relu";
        }
        public override Vol forward(Vol V, bool is_training) {
            this.in_act = V;
            var V2 = V.clone();
            var N = V.w.Length;
            var V2w = V2.w;
            for (var i = 0; i < N; i++)
            {
                if (V2w[i] < 0) V2w[i] = 0; // threshold at 0
            }
            this.out_act = V2;
            return this.out_act;
        }
        public override double backward(int y) {
            return 0;  
        }
        public override void backward() {
            var V = this.in_act; // we need to set dw of this
            var V2 = this.out_act;
            var N = V.w.Length;
            V.dw = Convnet_util.zeros(N); // zero out gradient wrt data
            for (var i = 0; i < N; i++)
            {
                if (V2.w[i] <= 0) V.dw[i] = 0; // threshold
                else V.dw[i] = V2.dw[i];
            }
            in_act = V;//check this
        }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }
    }

    class SigmoidLayer:layer
    {
        public SigmoidLayer(Layer_def opt)
        {
            // computed
            this.out_sx = opt.in_sx;
            this.out_sy = opt.in_sy;
            this.out_depth = opt.in_depth;
            this.layer_type = "sigmoid";
        }

        public override Vol forward(Vol V, bool is_training) {
            this.in_act = V;
            var V2 = V.cloneAndZero();
            var N = V.w.Length;
            var V2w = V2.w;
            var Vw = V.w;
            for (var i = 0; i < N; i++)
            {
                V2w[i] = 1.0 / (1.0 + Math.Exp(-Vw[i]));
            }
            this.out_act = V2;
            return this.out_act;
        }
        public override double backward(int y) { return 0; }
        public override void backward() {
            var V = this.in_act; // we need to set dw of this
            var V2 = this.out_act;
            var N = V.w.Length;
            V.dw = Convnet_util.zeros(N); // zero out gradient wrt data
            for (var i = 0; i < N; i++)
            {
                var v2wi = V2.w[i];
                V.dw[i] = v2wi * (1.0 - v2wi) * V2.dw[i];
            }
            in_act = V;//check this
        }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }
    }

    class MaxoutLayer : layer
    {

        int group_size;
        int [] switches;
        public MaxoutLayer(Layer_def opt)
        {
            this.group_size =  opt.group_size != 0 ? opt.group_size : 2;

            // computed
            this.out_sx = opt.in_sx;
            this.out_sy = opt.in_sy;
            this.out_depth =Convert.ToInt32( Math.Floor((double) opt.in_depth / this.group_size));
            this.layer_type = "maxout";
            switches = new int[out_depth];
        }

        public override Vol forward(Vol V, bool is_training)
        {
            this.in_act = V;
            var N       = this.out_depth;
            var V2      = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
           
            // optimization branch. If we're operating on 1D arrays we dont have
            // to worry about keeping track of x,y,d coordinates inside
            // input volumes. In convnets we do :(
            if (this.out_sx == 1 && this.out_sy == 1)
            {
                for (var i = 0; i < N; i++)
                {
                    var ix = i * this.group_size; // base index offset
                    var a = V.w[ix];
                    var ai = 0;
                    for (var j = 1; j < this.group_size; j++)
                    {
                        var a2 = V.w[ix + j];
                        if (a2 > a)
                        {
                            a = a2;
                            ai = j;
                        }
                    }
                    V2.w[i] = a;
                    this.switches[i] = ix + ai;
                }
            }
            else
            {
                var n = 0; // counter for switches
                for (var x = 0; x < V.sx; x++)
                {
                    for (var y = 0; y < V.sy; y++)
                    {
                        for (var i = 0; i < N; i++)
                        {
                            var ix = i * this.group_size;
                            var a = V.get(x, y, ix);
                            var ai = 0;
                            for (var j = 1; j < this.group_size; j++)
                            {
                                var a2 = V.get(x, y, ix + j);
                                if (a2 > a)
                                {
                                    a = a2;
                                    ai = j;
                                }
                            }
                            V2.set(x, y, i, a);
                            this.switches[n] = ix + ai;
                            n++;
                        }
                    }
                }

            }
            this.out_act = V2;
            return this.out_act;
        }
        public override double backward(int y) { return 0; }
        public override void backward()
        {
            var V = this.in_act; // we need to set dw of this
            var V2 = this.out_act;
            var N = this.out_depth;
            V.dw = Convnet_util.zeros(V.w.Length); // zero out gradient wrt data

            // pass the gradient through the appropriate switch
            if (this.out_sx == 1 && this.out_sy == 1)
            {
                for (var i = 0; i < N; i++)
                {
                    var chain_grad = V2.dw[i];
                    V.dw[this.switches[i]] = chain_grad;
                }
            }
            else
            {
                // bleh okay, lets do this the hard way
                var n = 0; // counter for switches
                for (var x = 0; x < V2.sx; x++)
                {
                    for (var y = 0; y < V2.sy; y++)
                    {
                        for (var i = 0; i < N; i++)
                        {
                            var chain_grad = V2.get_grad(x, y, i);
                            V.set_grad(x, y, this.switches[n], chain_grad);
                            n++;
                        }
                    }
                }
            }
            in_act = V;//check this
        }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }
    }

    class TanhLayer : layer
    {
        public TanhLayer(Layer_def opt)
        {
            this.out_sx = opt.in_sx;
            this.out_sy = opt.in_sy;
            this.out_depth = opt.in_depth;
            this.layer_type = "tanh";
        }

        public override Vol forward(Vol V, bool is_training)
        {
            this.in_act = V;
            var V2      = V.cloneAndZero();
            var N       = V.w.Length;
            for (var i = 0; i < N; i++)
            {               
                V2.w[i] = Math.Tanh(V.w[i]);
            }
            this.out_act = V2;
            return this.out_act;
        }
        public override double backward(int y) { return 0; }
        public override void backward()
        {
            var V = this.in_act; // we need to set dw of this
            var V2 = this.out_act;
            var N = V.w.Length;
            V.dw = Convnet_util.zeros(N); // zero out gradient wrt data
            for (var i = 0; i < N; i++)
            {
                var v2wi = V2.w[i];
                V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
            }
            in_act = V;//check this
        }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }

    }

}
