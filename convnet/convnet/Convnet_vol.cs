using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Drawing;
namespace convnet
{
    class Vol
    {
        public int sx { get; set; }
        public int sy { get; set; }
        public int depth { get; set; }
        public int n { get; set; }
        public double[] w;
        public double[] dw;

        public Vol(int sx, int sy, int depth)
        {
            this.sx = sx;
            this.sy = sy;
            this.depth = depth;
            n = sx * sy * depth;
            w = new double[n];
            dw = new double[n];

            var rndm = new Random();
            for (var i = 0; i < n; i++)
            {
                this.w[i] = rndm.NextDouble();
            }

        }

        public Vol(int sx, int sy, int depth,double c)
        {
            this.sx = sx;
            this.sy = sy;
            this.depth = depth;
            n = sx * sy * depth;
            w = new double[n];
            dw = new double[n];

            for (var i = 0; i < n; i++)
            {
                this.w[i] = c;
            }
        }

        public Vol(Bitmap bmp, bool convertGrayscale)
        {

        }

        public double get(int x,int y,int d)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            return this.w[ix];
        }

        public void set (int x,int y,int d,double v)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            this.w[ix] = v;
        }
        public void add(int x, int y, int d, double v)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            this.w[ix] += v;
        }
        public double get_grad(int x, int y, int d)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            return this.dw[ix];
        }
        public void set_grad(int x, int y, int d, double v)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            this.dw[ix] = v;
        }
        public void add_grad(int x, int y, int d, double v)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            this.dw[ix] += v;
        }
        public Vol cloneAndZero() { return new Vol(this.sx, this.sy, this.depth, 0.0); }
        public Vol clone()
        {
            var V = new Vol(this.sx, this.sy, this.depth, 0.0);
            var n = this.w.Length;
            for (var i = 0; i < n; i++) { V.w[i] = this.w[i]; }
            return V;
        }
        public void addFrom(Vol V) { for (var k = 0; k < this.w.Length; k++) { this.w[k] += V.w[k]; } }
        public void addFromScaled(Vol V, double a) { for (var k = 0; k < this.w.Length; k++) { this.w[k] += a * V.w[k]; } }
        public void setConst(double a) { for (var k = 0; k < this.w.Length; k++) { this.w[k] = a; } }

        

       //TO DO
        /*
        toJSON: function()
        {
            // todo: we may want to only save d most significant digits to save space
            var json = { }
      json.sx = this.sx;
            json.sy = this.sy;
            json.depth = this.depth;
            json.w = this.w;
            return json;
            // we wont back up gradients to save space
        },
    fromJSON: function(json)
        {
            this.sx = json.sx;
            this.sy = json.sy;
            this.depth = json.depth;

            var n = this.sx * this.sy * this.depth;
            this.w = global.zeros(n);
            this.dw = global.zeros(n);
            // copy over the elements.
            for (var i = 0; i < n; i++)
            {
                this.w[i] = json.w[i];
            }
        }
        */
    }
}
