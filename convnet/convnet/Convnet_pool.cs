using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class PoolLayer : layer
    {

        // required
        int sx;// filter size

        // optional
        int sy;
        int stride ;
        int pad ;
        
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        int [] switchx;
        int [] switchy;

        public PoolLayer(Layer_def opt)
        {            
            // required
            this.sx = opt.sx; // filter size
            this.in_depth = opt.in_depth;
            this.in_sx = opt.in_sx;
            this.in_sy = opt.in_sy;

            // optional
            this.sy =  opt.sy !=0 ? opt.sy : this.sx;
            this.stride =  opt.stride !=0 ? opt.stride : 2;
            this.pad =  opt.pad !=0 ? opt.pad : 0; // amount of 0 padding to add around borders of input volume

            // computed
            this.out_depth = this.in_depth;
            this.out_sx =Convert.ToInt32( Math.Floor( (double)  (this.in_sx + this.pad * 2 - this.sx) / this.stride + 1));
            this.out_sy =Convert.ToInt32( Math.Floor( (double)  (this.in_sy + this.pad * 2 - this.sy) / this.stride + 1));
            this.layer_type = "pool";
            // store switches for x,y coordinates for where the max comes from, for each output neuron
            this.switchx = Convnet_util.zeros_int(this.out_sx * this.out_sy * this.out_depth);
            this.switchy = Convnet_util.zeros_int(this.out_sx * this.out_sy * this.out_depth);

        }

        public override Vol forward(Vol V,bool is_training)
        {
            this.in_act = V;

            var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

            var n = 0; // a counter for switches
            for (var d = 0; d < this.out_depth; d++)
            {
                var x = -this.pad;
                var y = -this.pad;
                for (var ax = 0; ax < this.out_sx; x += this.stride, ax++)
                {
                    y = -this.pad;
                    for (var ay = 0; ay < this.out_sy; y += this.stride, ay++)
                    {

                        // convolve centered at this particular location
                        double a = -99999; // hopefully small enough ;\
                        int winx = -1, winy = -1;
                        for (var fx = 0; fx < this.sx; fx++)
                        {
                            for (var fy = 0; fy < this.sy; fy++)
                            {
                                var oy = y + fy;
                                var ox = x + fx;
                                if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx)
                                {
                                    var v = V.get(ox, oy, d);
                                    // perform max pooling and store pointers to where
                                    // the max came from. This will speed up backprop 
                                    // and can help make nice visualizations in future
                                    if (v > a) { a = v; winx = ox; winy = oy; }
                                }
                            }
                        }
                        this.switchx[n] = winx;
                        this.switchy[n] = winy;
                        n++;
                        A.set(ax, ay, d, a);
                    }
                }
            }
            this.out_act = A;
            return this.out_act;
        }
        public override void backward()
        {
            // pooling layers have no parameters, so simply compute 
            // gradient wrt data here
            var V = this.in_act;
            V.dw = Convnet_util.zeros(V.w.Length); // zero out gradient wrt data
            var A = this.out_act; // computed in forward pass 

            var n = 0;
            for (var d = 0; d < this.out_depth; d++)
            {
                var x = -this.pad;
                var y = -this.pad;
                for (var ax = 0; ax < this.out_sx; x += this.stride, ax++)
                {
                    y = -this.pad;
                    for (var ay = 0; ay < this.out_sy; y += this.stride, ay++)
                    {

                        var chain_grad = this.out_act.get_grad(ax, ay, d);
                        V.add_grad(this.switchx[n], this.switchy[n], d, chain_grad);
                        n++;

                    }
                }
            }
        }
        public override double backward(int y) { return 0; }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }
    }
}
