using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{

    class LocalResponseNormalizationLayer : layer
    {
        int k, n;
        double alpha, beta;
        Vol S_cache_;        
        public LocalResponseNormalizationLayer(Layer_def opt)
        {
            // required
            this.k = opt.k;
            this.n = opt.n;
            this.alpha = opt.alpha;
            this.beta = opt.beta;

            // computed
            this.out_sx = opt.in_sx;
            this.out_sy = opt.in_sy;
            this.out_depth = opt.in_depth;
            this.layer_type = "lrn";

            // checks
            if (this.n % 2 == 0) { Console.WriteLine("WARNING n should be odd for LRN layer"); }
        }
        public override Vol forward(Vol V, bool is_training)
        {
            this.in_act = V;

            var A = V.cloneAndZero();
            this.S_cache_ = V.cloneAndZero();
            var n2 = Math.Floor((double) this.n / 2);
            for (var x = 0; x < V.sx; x++)
            {
                for (var y = 0; y < V.sy; y++)
                {
                    for (var i = 0; i < V.depth; i++)
                    {

                        var ai = V.get(x, y, i);

                        // normalize in a window of size n
                        var den = 0.0;
                        for (var j = Math.Max(0, i - n2); j <= Math.Min(i + n2, V.depth - 1); j++)
                        {
                            var aa = V.get(x, y, (int)j);
                            den += aa * aa;
                        }
                        den *= this.alpha / this.n;
                        den += this.k;
                        this.S_cache_.set(x, y, i, den); // will be useful for backprop
                        den = Math.Pow(den, this.beta);
                        A.set(x, y, i, ai / den);
                    }
                }
            }

            this.out_act = A;
            return this.out_act; // dummy identity function for now
        }
        public override double backward(int y)
        {
            return 0;
        }
        public override void backward()
        {
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
}
