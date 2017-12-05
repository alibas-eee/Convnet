using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
namespace convnet
{  
    class ConvLayer:layer
    {
        // required
        //int out_depth;
        int sx; // filter size. Should be odd if possible, it's cleaner.
        int sy; 
        int stride;
        int pad; 
        //List<Vol> filters;

        public ConvLayer(Layer_def opt)
        { 
                // required
            out_depth = opt.filters;
            sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
            in_depth = opt.in_depth;
            in_sx = opt.in_sx;
            in_sy = opt.in_sy;
    
            // optional
            this.sy = opt.sy != 0 ? opt.sy : this.sx;
            this.stride =  opt.stride != 0 ? opt.stride : 1; // stride at which we apply filters to input volume
            this.pad =  opt.pad != 0 ? opt.pad : 0; // amount of 0 padding to add around borders of input volume
            this.l1_decay_mul =  opt.l1_decay_mul != 0 ? opt.l1_decay_mul : 0.0;
            this.l2_decay_mul =  opt.l2_decay_mul != 0 ? opt.l2_decay_mul : 1.0;

            // computed
            // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
            // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
            // final application.
            this.out_sx =Convert.ToInt32( Math.Floor((double)  (this.in_sx + this.pad * 2 - this.sx) / this.stride + 1));
            this.out_sy =Convert.ToInt32( Math.Floor((double)  (this.in_sy + this.pad * 2 - this.sy) / this.stride + 1));
            this.layer_type = "conv";

            
            // initializations
            var bias = opt.bias_pref != 0 ? opt.bias_pref : 0.0;
            this.filters = new List<Vol>();
            for(var i=0;i<this.out_depth;i++) { this.filters.Add(new Vol(this.sx, this.sy, this.in_depth)); }
            this.biases = new Vol(1, 1, this.out_depth, bias);



        }
        public override Vol forward(Vol V,bool is_training) {
          // optimized code by @mdda that achieves 2x speedup over previous version

          this.in_act = V;
          var A = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
      
          var V_sx = V.sx |0;
          var V_sy = V.sy |0;
          var xy_stride = this.stride |0;

          for(var d=0;d<this.out_depth;d++) {
            var f = this.filters[d];
            var x = -this.pad |0;
            var y = -this.pad |0;
            for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
              x = -this.pad |0;
              for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

                // convolve centered at this particular location
                var a = 0.0;
                for(var fy=0;fy<f.sy;fy++) {
                  var oy = y+fy; // coordinates in the original input array coordinates
                  for(var fx=0;fx<f.sx;fx++) {
                    var ox = x+fx;
                    if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                      for(var fd=0;fd<f.depth;fd++) {
                        // avoid function call overhead (x2) for efficiency, compromise modularity :(
                        a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd];
                      }
                    }
                  }
                }
                a += this.biases.w[d];
                A.set(ax, ay, d, a);
              }
            }
          }
          this.out_act = A;
          return this.out_act;
        }
        public override void backward() {

          var V = this.in_act;
          V.dw = Convnet_util.zeros(V.w.Length); // zero out gradient wrt bottom data, we're about to fill it

          var V_sx = V.sx |0;
          var V_sy = V.sy |0;
          var xy_stride = this.stride |0;

          for(var d=0;d<this.out_depth;d++) {
            var f = this.filters[d];
            var x = -this.pad |0;
            var y = -this.pad |0;
            for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
              x = -this.pad |0;
              for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

                // convolve centered at this particular location
                var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
                for(var fy=0;fy<f.sy;fy++) {
                  var oy = y+fy; // coordinates in the original input array coordinates
                  for(var fx=0;fx<f.sx;fx++) {
                    var ox = x+fx;
                    if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                      for(var fd=0;fd<f.depth;fd++) {
                        // avoid function call overhead (x2) for efficiency, compromise modularity :(
                        var ix1 = ((V_sx * oy)+ox)*V.depth+fd;
                        var ix2 = ((f.sx * fy)+fx)*f.depth+fd;
                        f.dw[ix2] += V.w[ix1]*chain_grad;
                        V.dw[ix1] += f.w[ix2]*chain_grad;
                      }
                    }
                  }
                }
                this.biases.dw[d] += chain_grad;
              }
            }
          }
        }
        public override double backward(int y) { return 0; }

        public override List<params_and_grads> getParamsAndGrads()
        {
            params_and_grads p_g=new  params_and_grads();
            List<params_and_grads> response = new List<params_and_grads>();
            for (var i = 0; i < this.out_depth; i++)
            {
                p_g = new params_and_grads();
                p_g.param = filters[i].w;
                p_g.grads = filters[i].dw;
                p_g.l1_decay_mul = l1_decay_mul;
                p_g.l2_decay_mul = l2_decay_mul;
                response.Add(p_g);
            }
            
            p_g = new params_and_grads();
            p_g.param = biases.w;
            p_g.grads = biases.dw;
            p_g.l1_decay_mul = 0;
            p_g.l2_decay_mul = 0;
            response.Add(p_g);
            return response;
        }
    }

    class FullyConnLayer:layer
    {        
        public FullyConnLayer(Layer_def opt)
        {
            // required
            // ok fine we will allow 'filters' as the word as well
            this.out_depth =  opt.num_neurons !=0 ? opt.num_neurons : opt.filters;

            // optional 
            this.l1_decay_mul =  opt.l1_decay_mul !=0 ? opt.l1_decay_mul : 0.0;
            this.l2_decay_mul =  opt.l2_decay_mul !=0 ? opt.l2_decay_mul : 1.0;

            // computed
            this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
            this.out_sx = 1;
            this.out_sy = 1;
            this.layer_type = "fc";

            // initializations
            var bias =  opt.bias_pref !=0 ? opt.bias_pref : 1.0;
            this.filters = new List<Vol>();
            for(var i=0;i<this.out_depth ;i++) { this.filters.Add(new Vol(1, 1, this.num_inputs)); }
            this.biases = new Vol(1, 1, this.out_depth, bias);                    
        }
        public override Vol forward(Vol V, bool is_training)
        {
            this.in_act = V;
            var A = new Vol(1, 1, this.out_depth, 0.0);
            var Vw = V.w;
            for (var i = 0; i < this.out_depth; i++)
            {
                var a = 0.0;
                var wi = this.filters[i].w;
                for (var d = 0; d < this.num_inputs; d++)
                {
                    a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
                }
                a += this.biases.w[i];
                A.w[i] = a;
            }
            this.out_act = A;
            return this.out_act;
        }
        public override void backward()
        {
            var V = this.in_act;
            V.dw = Convnet_util.zeros(V.w.Length); // zero out the gradient in input Vol

            // compute gradient wrt weights and data
            for (var i = 0; i < this.out_depth; i++)
            {
                var tfi = this.filters[i];
                var chain_grad = this.out_act.dw[i];
                for (var d = 0; d < this.num_inputs; d++)
                {
                    V.dw[d] += tfi.w[d] * chain_grad; // grad wrt input data
                    tfi.dw[d] += V.w[d] * chain_grad; // grad wrt params
                }
                this.biases.dw[i] += chain_grad;
            }
        }
        public override double backward(int y) { return 0; }
        
        public override List<params_and_grads> getParamsAndGrads()
        {
            params_and_grads p_g = new params_and_grads();
            List<params_and_grads> response = new List<params_and_grads>();
            for (var i = 0; i < this.out_depth; i++)
            {
                p_g = new params_and_grads();
                p_g.param = filters[i].w;
                p_g.grads = filters[i].dw;
                p_g.l1_decay_mul = l1_decay_mul;
                p_g.l2_decay_mul = l2_decay_mul;
                response.Add(p_g);
            }

            p_g = new params_and_grads();
            p_g.param = biases.w;
            p_g.grads = biases.dw;
            p_g.l1_decay_mul = 0;
            p_g.l2_decay_mul = 0;
            response.Add(p_g);
            return response;
        }
    }
}
