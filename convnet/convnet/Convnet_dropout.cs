using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class DropoutLayer : layer
    {
        double drop_prob;
        bool[] dropped;      
        Random rnd;
        public DropoutLayer(Layer_def opt)
        {
            // computed
            this.out_sx = opt.in_sx;
            this.out_sy = opt.in_sy;
            this.out_depth = opt.in_depth;
            this.layer_type = "dropout";
            this.drop_prob = 0.5;
            this.dropped = Convnet_util.zeros_bool(this.out_sx * this.out_sy * this.out_depth);
            rnd = new Random();
        }
        public override Vol forward(Vol V, bool is_training)
        {
            this.in_act = V;
            var V2 = V.clone();
            var N = V.w.Length;
            if (is_training)
            {
                // do dropout
                for (var i = 0; i < N; i++)
                {
                    if (rnd.NextDouble() < this.drop_prob) { V2.w[i] = 0; this.dropped[i] = true; } // drop!
                    else { this.dropped[i] = false; }
                }
            }
            else
            {
                // scale the activations during prediction
                for (var i = 0; i < N; i++) { V2.w[i] *= this.drop_prob; }
            }
            this.out_act = V2;
            return this.out_act; // dummy identity function for now
        }
        public override double backward(int y)
        {
            return 0;
        }
        public override void backward()
        {
            var V = this.in_act; // we need to set dw of this
            var chain_grad = this.out_act;
            var N = V.w.Length;
            V.dw = Convnet_util.zeros(N); // zero out gradient wrt data
            for (var i = 0; i < N; i++)
            {
                if (!(this.dropped[i]))
                {
                    V.dw[i] = chain_grad.dw[i]; // copy over the gradient
                }
            }
            in_act = V;//check this
        }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }
        
    }
}
