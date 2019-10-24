/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "lmptype.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include "displace_atoms.h"
#include "atom.h"
#include "modify.h"
#include "domain.h"
#include "lattice.h"
#include "comm.h"
#include "irregular.h"
#include "group.h"
#include "math_const.h"
#include "random_park.h"
#include "error.h"
#include "force.h"

#include "thermo.h"

#include "math.h"
#include <algorithm>
#include "mpi.h"
#include <sstream>
#include <string>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <istream>

using namespace LAMMPS_NS;
using namespace MathConst;

using namespace std;


enum{MOVE,MOVE1,RAMP,RANDOM,EDGE,SCREW,LOOP,ROTATE,SEDGE,SSCREW,LOOPSIZE,STRAIGHT,BOUNDARY,BOUNDARY1};

/* ---------------------------------------------------------------------- */

DisplaceAtoms::DisplaceAtoms(LAMMPS *lmp) : Pointers(lmp) {}

/* ---------------------------------------------------------------------- */

void DisplaceAtoms::command(int narg, char **arg)
{
  int i,j,k,l;

  if (domain->box_exist == 0)
    error->all(FLERR,"Displace_atoms command before simulation box is defined");
  if (narg < 2) error->all(FLERR,"Illegal displace_atoms command");
  if (modify->nfix_restart_peratom)
    error->all(FLERR,"Cannot displace_atoms after "
               "reading restart file with per-atom info");

  if (comm->me == 0 && screen) fprintf(screen,"Displacing atoms ...");

  // group and style

  int igroup = group->find(arg[0]);
  if (igroup == -1) error->all(FLERR,"Could not find displace_atoms group ID");
  int groupbit = group->bitmask[igroup];

  int style;
  if (strcmp(arg[1],"move") == 0) style = MOVE;
  else if (strcmp(arg[1],"ramp") == 0) style = RAMP;
  else if (strcmp(arg[1],"move1") == 0) style = MOVE1;
  else if (strcmp(arg[1],"random") == 0) style = RANDOM;
  else if (strcmp(arg[1],"edge") == 0) style = EDGE;
  else if (strcmp(arg[1],"screw") == 0) style = SCREW;
  else if (strcmp(arg[1],"loop") == 0) style = LOOP;
  else if (strcmp(arg[1],"rotate") == 0) style = ROTATE;
  else if (strcmp(arg[1],"sedge") == 0) style = SEDGE;
  else if (strcmp(arg[1],"sscrew") == 0) style = SSCREW;
  else if (strcmp(arg[1],"loopsize") == 0) style = LOOPSIZE;
  else if (strcmp(arg[1],"straight") == 0) style = STRAIGHT;
  else if (strcmp(arg[1],"boundary") == 0) style = BOUNDARY;
  else if (strcmp(arg[1],"boundary1") == 0) style = BOUNDARY1;
  else error->all(FLERR,"Illegal displace_atoms command/type");

  // set option defaults

  scaleflag = 1;

  // read options from end of input line

  if (style == MOVE) options(narg-5,&arg[5]);
  else if (style == MOVE1) options(narg-2,&arg[2]);
  else if (style == RAMP) options(narg-8,&arg[8]);
  else if (style == RANDOM) options(narg-6,&arg[6]);
  else if (style == EDGE) options(narg-13,&arg[13]);
  else if (style == SCREW) options(narg-12,&arg[12]);
  else if (style == LOOP) options(narg-24,&arg[24]);
  else if (style == ROTATE) options(narg-9,&arg[9]);
  else if (style == SEDGE) options(narg-6,&arg[6]);
  else if (style == LOOPSIZE) options(narg-18,&arg[18]);
  else if (style == SSCREW) options(narg-5,&arg[5]);
  else if (style == STRAIGHT) options(narg-8,&arg[8]);
  else if (style == BOUNDARY) options(narg-5,&arg[5]);
  else if (style == BOUNDARY1) options(narg-5,&arg[5]);
  // setup scaling

  double xscale,yscale,zscale;
  if (scaleflag) {
    xscale = domain->lattice->xlattice;
    yscale = domain->lattice->ylattice;
    zscale = domain->lattice->zlattice;
  }
  else xscale = yscale = zscale = 1.0;

  // move atoms bv[1] 3-vector

  if (style == MOVE) {

    double delx = xscale*force->numeric(FLERR,arg[2]);
    double dely = yscale*force->numeric(FLERR,arg[3]);
    double delz = zscale*force->numeric(FLERR,arg[4]);

    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        x[i][0] += delx;
        x[i][1] += dely;
        x[i][2] += delz;
      }
    }
  }
  
   // move atoms bv[1] 3-vector

  if (style == MOVE1) {
  
	double **x = atom->x;
	int *mask = atom->mask;
    int nlocal = atom->nlocal;
	tagint *tag = atom->tag;
	int my_rank;
	
//	MPI_Init(&narg, &arg);
	
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	
	
		int number_of_lines = 0;

		ifstream fin("dump.disp");
		string line;
	
		//count number of lines
		while (getline(fin,line))
		{
			number_of_lines = number_of_lines + 1;
		}
		int noa=number_of_lines/2-9;
		
		//cout<<"number of atoms"<<noa<<endl;
		int start = noa+19;
		int end = (noa+9)*2+1;
		double dispx[noa],dispy[noa],dispz[noa],tempx,tempy,tempz;
		unsigned int count = 0;
	
		int temp;
		fin.clear( );
		fin.seekg( 0, std::ios::beg );
	
		
	if (my_rank == 0)
	{
		while (getline(fin,line))
		{
		
			count++;
			if (count > end) { break; }    // done
			if (count < start)  { continue; } // too early
	
			istringstream iss(line);
		
			iss >> temp; 
			iss >> tempx;
			iss >> tempy;
			iss >> tempz;
		
			
			int j = temp -1;
			dispx[j]=tempx;
			dispy[j]=tempy;
			dispz[j]=tempz;
			//cout<<endl<<"j "<<temp<<"dispx[j]"<<dispx[j]<<"dispy[j] "<<dispy[j]<<"dispz[j]"<<dispz[j];
		}
		fin.close();
	}	
	 
	
		MPI_Bcast(dispx,noa,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(dispy,noa,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(dispz,noa,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	
	
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
		int itag = tag[i]-1;
		//cout<<endl<<"i: "<<i<<"i[tag]"<<i[tag]<<"x: "<<x[i][0]<<"y: "<<x[i][1]<<"z: "<<x[i][2]<<"dispx[i]"<<dispx[itag]<<"dispy[i] "<<dispy[itag]<<"dispz[i] "<<dispz[itag];
		
        x[i][0] += dispx[itag];
        x[i][1] += dispy[itag];
        x[i][2] += dispz[itag];
      }
    }
  }
  

  // move atoms in ramped fashion

  if (style == RAMP) {

    int d_dim;
    if (strcmp(arg[2],"x") == 0) d_dim = 0;
    else if (strcmp(arg[2],"y") == 0) d_dim = 1;
    else if (strcmp(arg[2],"z") == 0) d_dim = 2;
    else error->all(FLERR,"Illegal displace_atoms ramp command");

    double d_lo,d_hi;
    if (d_dim == 0) {
      d_lo = xscale*force->numeric(FLERR,arg[3]);
      d_hi = xscale*force->numeric(FLERR,arg[4]);
    } else if (d_dim == 1) {
      d_lo = yscale*force->numeric(FLERR,arg[3]);
      d_hi = yscale*force->numeric(FLERR,arg[4]);
    } else if (d_dim == 2) {
      d_lo = zscale*force->numeric(FLERR,arg[3]);
      d_hi = zscale*force->numeric(FLERR,arg[4]);
    }

    int coord_dim;
    if (strcmp(arg[5],"x") == 0) coord_dim = 0;
    else if (strcmp(arg[5],"y") == 0) coord_dim = 1;
    else if (strcmp(arg[5],"z") == 0) coord_dim = 2;
    else error->all(FLERR,"Illegal displace_atoms ramp command");

    double coord_lo,coord_hi;
    if (coord_dim == 0) {
      coord_lo = xscale*force->numeric(FLERR,arg[6]);
      coord_hi = xscale*force->numeric(FLERR,arg[7]);
    } else if (coord_dim == 1) {
      coord_lo = yscale*force->numeric(FLERR,arg[6]);
      coord_hi = yscale*force->numeric(FLERR,arg[7]);
    } else if (coord_dim == 2) {
      coord_lo = zscale*force->numeric(FLERR,arg[6]);
      coord_hi = zscale*force->numeric(FLERR,arg[7]);
    }

    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    double fraction,dramp;

    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        fraction = (x[i][coord_dim] - coord_lo) / (coord_hi - coord_lo);
        fraction = MAX(fraction,0.0);
        fraction = MIN(fraction,1.0);
        dramp = d_lo + fraction*(d_hi - d_lo);
        x[i][d_dim] += dramp;
      }
    }
  }

  // move atoms randomly
  // makes atom result independent of what proc owns it via random->reset()

  if (style == RANDOM) {
    RanPark *random = new RanPark(lmp,1);

    double dx = xscale*force->numeric(FLERR,arg[2]);
    double dy = yscale*force->numeric(FLERR,arg[3]);
    double dz = zscale*force->numeric(FLERR,arg[4]);
    int seed = force->inumeric(FLERR,arg[5]);
    if (seed <= 0) error->all(FLERR,"Illegal displace_atoms random command");

    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        random->reset(seed,x[i]);
        x[i][0] += dx * 2.0*(random->uniform()-0.5);
        x[i][1] += dy * 2.0*(random->uniform()-0.5);
        x[i][2] += dz * 2.0*(random->uniform()-0.5);
      }
    }

    delete random;
  }
  
  // move boundary atoms to evenly space them for straight dislocation generation algorithm

  if (style == BOUNDARY) {
  

    double b = force->numeric(FLERR,arg[2]);
    double lz = force->numeric(FLERR,arg[3]);
    double theta = force->numeric(FLERR,arg[4]);
//	double dz = force -> numeric(FLERR,arg[5]);

    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

	double ux,uy,uz;
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        
		ux = (- copysign(0.5,x[i][1]) * b * cos(theta) * (x[i][2] + lz / 2))/ lz + b * cos(theta)/2; 
		uy = 0;
		uz = (- copysign(0.5,x[i][1]) * b * sin(theta) * (x[i][2] + lz / 2))/ lz + b * sin(theta)/2;
	
        x[i][0] += ux;
        x[i][1] += uy;
        x[i][2] += uz;
      }
    }

  }
  
  // move boundary atoms to evenly space them for straight dislocation generation algorithm

  if (style == BOUNDARY1) {
  

    double b = force->numeric(FLERR,arg[2]);
    double lz = force->numeric(FLERR,arg[3]);
    double theta = force->numeric(FLERR,arg[4]);
//    double dz = force -> numeric(FLERR,arg[5]);

    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

	double ux,uy,uz;
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (x[i][1]>0)	{
		ux = (- b * cos(theta) / 4) * (x[i][2] / (lz / 2)); 
		uy = 0;
		uz = (- b * sin(theta) / 4) * (x[i][2] / (lz / 2)); 
		}
		else
		{
		ux = ( b * cos(theta) / 4) * (x[i][2] / (lz / 2 - b * cos(theta) /2 )); 
		uy = 0;
		uz = ( b * sin(theta) / 4) * (x[i][2] / (lz / 2 - b * sin(theta) /2 ));	
		}
		
        x[i][0] += ux;
        x[i][1] += uy;
        x[i][2] += uz;
      }
    }

  }
  // move atoms to generate an infinite straight edge dislocation
  // P = point = vector = core of the dislocation
  

if (style == EDGE) {
 
	
	
	double y0 = yscale*force->numeric(FLERR,arg[2]);
    double z0 = zscale*force->numeric(FLERR,arg[3]);
	double ly = yscale*force->numeric(FLERR,arg[4]);
	double lz = zscale*force->numeric(FLERR,arg[5]);
    double b = force->numeric(FLERR,arg[6]);
    //if (b <= 0) error->all(FLERR,"Illegal Burger's value");
	double angle = force->numeric(FLERR,arg[7]);
	double v = force->numeric(FLERR,arg[8]);
	int ny = force->numeric(FLERR,arg[9]);	
	int nz = force->numeric(FLERR,arg[10]);
	double yeva = force->numeric(FLERR,arg[11]);	
	double zeva = force->numeric(FLERR,arg[12]);	
	
	
    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
	
	
	double znpos,znneg,ynpos,ynneg,sy[2],sz[2],u0[3],uy[3],uz[3],y[nlocal][3],d[3],theta;
	
	u0[0] = 0;
	u0[1] = 0;
	uy[0] = 0;
	uy[1] = 0;
	uz[0] = 0;
	uz[1] = 0;
	for (j = 0; j < nz; j++) {
		for (k = 0 ; k < ny; k++) {
			znpos = zeva - (z0 + (j - ((nz - 1) / 2)) * lz );
			ynpos = yeva - (y0 + (k - ((ny - 1) / 2)) * ly );
			
			
			
			u0[1] = u0[1] - (b /(2 * MY_PI)) * (((1 - 2 * v) / ( 4 * (1 - v))) * log(znpos * znpos + ynpos * ynpos) + (znpos * znpos - ynpos * ynpos) / (4 * (1 - v) * (znpos * znpos + ynpos * ynpos)));
			u0[2] = u0[2] + (b /(2 * MY_PI)) * ( theta + ((znpos * ynpos) / ( 2 * (1 - v) * (znpos * znpos + ynpos * ynpos)))); 
			
			uy[1] = uy[1] - (b /(2 * MY_PI)) * (((1 - 2 * v) / ( 4 * (1 - v))) * log(znpos * znpos + (ynpos + ly) * (ynpos + ly)) + (znpos * znpos - (ynpos + ly) * (ynpos + ly)) / (4 * (1 - v) * (znpos * znpos + (ynpos + ly) * (ynpos + ly)))) ; 
			uy[2] = uy[2] - (b /(2 * MY_PI)) * (((1 - 2 * v) / ( 4 * (1 - v))) * log((znpos + lz) * (znpos + lz) + ynpos * ynpos) + ((znpos + lz) * (znpos + lz) - ynpos * ynpos) / (4 * (1 - v) * ((znpos + lz) * (znpos + lz) + ynpos * ynpos))) ;
			
			uz[1] = uz[1] + (b /(2 * MY_PI)) * ( theta + atan((ynpos + ly)/znpos) + ((znpos * (ynpos + ly)) / ( 2 * (1 - v) * (znpos * znpos + (ynpos + ly) * (ynpos + ly))))) ;  
			uz[2] = uz[2] + (b /(2 * MY_PI)) * ( theta + (((znpos + lz) * ynpos) / ( 2 * (1 - v) * ((znpos + lz) * (znpos + lz) + ynpos * ynpos)))) ; 	
								
				}		
			}	
	
	sz[0] = (uz[2] - u0[0]) / lz;
	sz[1] = (uz[1] - u0[0]) / ly;
				
	sy[0] = (uy[2] - u0[1]) / lz;
	sy[1] = (uy[1] - u0[1]) / ly;
							
	cout << " sz[0] = ";
	cout << sz[0];
	cout << " sz[1] = ";
	cout << sz[1];
	cout << " sy[0] = ";
	cout << sy[0];
	cout << " sy[1] = ";
	cout << sy[1];
	
			// z0 is the z position of the dislocation core
			// y0 is the y position of the dislocation core

	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
		y[i][1] = 0;
		y[i][2] = 0;
		for (j = 0; j < nz; j++) {
			for (k = 0 ; k < ny; k++) {
				//znpos = x[i][2]- (z0 + (j - ((nz - 1) / 2)) * lz );
				
				//ynpos = x[i][1] - (y0 + (k - ((ny - 1) / 2)) * ly );
				znpos = x[i][2] - (z0);
				ynpos = x[i][1] - (y0);
				
			//	if (atan(ynpos/znpos)>=0) {
			//		theta = atan(ynpos/znpos);
			//	}
			//	if ((znpos<0) && (ynpos==0)) {
			//		theta = MY_PI/2;
			//	}
			//	if (atan(ynpos/znpos)<0) {
			//		theta = atan(ynpos/znpos)+MY_PI;
			//	}
				theta = atan(ynpos/znpos);
				y[i][1] = y[i][1] - (b /(2 * MY_PI)) * (((1 - 2 * v) / ( 4 * (1 - v))) * log(znpos * znpos + ynpos * ynpos) + (znpos * znpos - ynpos * ynpos) / (4 * (1 - v) * (znpos * znpos + ynpos * ynpos)));  
				y[i][2] = y[i][2] + (b /(2 * MY_PI)) * ( theta + ((znpos * ynpos) / ( 2 * (1 - v) * (znpos * znpos + ynpos * ynpos)))) ;
		
		}	
		}
		
		if (nz==1){
		d[1] = y[i][1];
		d[2] = y[i][2];
		
		}
		else
		{
		d[1] = y[i][1] - (sy[0] * x[i][2] + sy[1] * x[i][1]);
		d[2] = y[i][2] - (sz[0] * x[i][2] + sz[1] * x[i][1]);
		}
		
		x[i][1] += d[1];
		x[i][2] += d[2];
		
	//	znpos = x[i][2] - (z0);
	//	ynpos = x[i][1] - (y0);
	//	x[i][1] += - (b /(2 * MY_PI)) * (((1 - 2 * v) / ( 4 * (1 - v))) * log(znpos * znpos + ynpos * ynpos) + (znpos * znpos - ynpos * ynpos) / (4 * (1 - v) * (znpos * znpos + ynpos * ynpos)));
	//	x[i][2] += (b /(2 * MY_PI)) * ( theta + ((znpos * ynpos) / ( 2 * (1 - v) * (znpos * znpos + ynpos * ynpos)))) ;
		
	}	
	}		
 }

  
 // move atoms to generate an infinite straight screw dislocation
  // P = point = vector = core of the dislocation
  
  if (style == SCREW) {
  
	double x0 = xscale*force->numeric(FLERR,arg[2]);
    double y0 = yscale*force->numeric(FLERR,arg[3]);
	double a = force->numeric(FLERR,arg[4]);
	//if (a < 0) error->all(FLERR,"Illegal half distance between dislocation core");
	double b = force->numeric(FLERR,arg[5]);
    //if (b <= 0) error->all(FLERR,"Illegal Burger's value");
    double v = force->numeric(FLERR,arg[6]);
	int nx = force->numeric(FLERR,arg[7]);	
	int ny = force->numeric(FLERR,arg[8]);
	double lx = force->numeric(FLERR,arg[9]);
	double ly = force->numeric(FLERR,arg[10]);
	double xlow = force->numeric(FLERR,arg[11]);
	double ylow = force->numeric(FLERR,arg[12]);
	
    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
	
	double y[nlocal][2],sx[2],sy[2],u,u0[2],uz[2],uy[2], yl, xlpos,xlneg,d[2],sz[2];
	
	u = 0;
	uz[0] = 0;
	uz[1] = 0;
	
	for (j = 0; j < nx; j++) {
		for (k = 0 ; k < ny; k++) {
			xlpos = xlow - (x0 + a + (j - ((nx - 1) / 2)) * lx );
			xlneg = xlow - (x0 - a + (k - ((nx - 1) / 2)) * lx );
			yl = ylow - (y0 + (k - ((ny - 1) / 2)) * ly );	
			
			u = u + b * (atan2(yl,xlpos)) / (2 * MY_PI) - b * (atan2(yl,xlneg)) / (2 * MY_PI);
						
			uz[0] = uz[0] + b * (atan2(yl,xlpos+lx)) / (2 * MY_PI) - b * (atan2(yl,xlneg+lx)) / (2 * MY_PI);						
			uz[1] = uz[1] + b * (atan2(yl+ly,xlpos)) / (2 * MY_PI) - b * (atan2(yl+ly,xlneg)) / (2 * MY_PI);
					}
				}	
		
				sz[0] = (uz[0] - u) / lx;
				sz[1] = (uz[1] - u) / ly;
	cout << " sz[0] = ";
	cout << sz[0];
	cout << " sz[1] = ";
	cout << sz[1];
	
	cout << " nlocal = " ;
	cout << nlocal;
	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
		y[i][0] = 0;
		for (j = 0; j < nx; j++) {
			for (k = 0 ; k < ny; k++) {
				xlpos = x[i][0] - (x0 + a + (j - ((nx - 1) / 2)) * lx );
				xlneg = x[i][0] - (x0 - a + (k - ((nx - 1) / 2)) * lx );
				yl = x[i][1] - (y0 + (k - ((ny - 1) / 2)) * ly );	
				
				y[i][0] = y[i][0] + b * (atan2(yl,xlpos)) / (2 * MY_PI) - b * (atan2(yl,xlneg)) / (2 * MY_PI);
				
			}
		}
		
		x[i][2] += y[i][0] - (sz[0] * (x[i][0]) + sz[1] * (x[i][1]));
		
    }
  }
  }
  



     // move atoms to generate a mixed/edge/screw straight dislocation
  // P = point = vector = core of the dislocation
  
  if (style == STRAIGHT) {
 
	
	
	double y0 = yscale*force->numeric(FLERR,arg[2]);
    double z0 = zscale*force->numeric(FLERR,arg[3]);
	double b = force->numeric(FLERR,arg[4]);
	double angle = force->numeric(FLERR,arg[5]);
	double v = force->numeric(FLERR,arg[6]);
	//flag  to turn on the boundary displacement field
	double flag = force->numeric(FLERR,arg[7]);
	
	double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
	
	
	double	yn,zn,theta;
	
	// z0 is the z position of the dislocation core
	// y0 is the y position of the dislocation core

	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
		
			//Compute relative position of current atom
			zn = x[i][2] - z0;
			yn = x[i][1] - y0;
			
			if (zn<0) {
				theta = atan(yn/zn)+MY_PI;
			}
			if ((zn>0) && (yn>=0)) {
				theta = atan(yn/zn);
			}
			if ((zn>0) && (yn<0)) {
				theta = atan(yn/zn)+ 2*MY_PI;
			}
			if ((zn==0) && (yn>0)) {           
                theta =MY_PI/2;
            }
			if ((zn==0) && (yn<0)) { 
                theta =3*MY_PI/2;                 
            }
			if ((zn==0) && (yn==0)) {
				error->warning(FLERR,"atan(0/0)");
                theta =-1000000000;
            }
			if (flag == 0){
			//Volterra displacement field
			x[i][0] += cos (angle) * (b /(2 * MY_PI)) * theta;
			x[i][1] += - sin (angle) * (b /(2 * MY_PI)) * (((1 - 2 * v) / ( 4 * (1 - v))) * log(zn * zn + yn * yn) + (zn * zn - yn * yn) / (4 * (1 - v) * (zn * zn + yn * yn)));
			x[i][2] += sin (angle) * (b /(2 * MY_PI)) * ( theta + ((zn * yn) / ( 2 * (1 - v) * (zn * zn + yn * yn))));
			}
			
			else {
			//Volterra displacement field
			x[i][0] += cos (angle) * (b /(2 * MY_PI) * theta);
			x[i][1] += 0;
			x[i][2] += sin (angle) * (b /(2 * MY_PI)) * ( theta + ((zn * yn) / ( 2 * (1 - v) * (zn * zn + yn * yn))));
			}
	}	
	}		
					
		
}	
  
  
  
  
  



  
  
  // move atoms to generate a dislocation loop
  // P = point = vector = core of the dislocation
  
  if (style == LOOP) {
  
	double x1 = force->numeric(FLERR,arg[2]);
    double x2 = force->numeric(FLERR,arg[3]);
    double x3 = force->numeric(FLERR,arg[4]);
	double y1 = force->numeric(FLERR,arg[5]);
    double y2 = force->numeric(FLERR,arg[6]);
    double y3 = force->numeric(FLERR,arg[7]);
	double z1 = force->numeric(FLERR,arg[8]);
    double z2 = force->numeric(FLERR,arg[9]);
    double z3 = force->numeric(FLERR,arg[10]);
	
	double x0 = xscale*force->numeric(FLERR,arg[11]);
    double y0 = yscale*force->numeric(FLERR,arg[12]);
    double z0 = zscale*force->numeric(FLERR,arg[13]);
    double b = force->numeric(FLERR,arg[14]);
    if (b <= 0) error->all(FLERR,"Illegal Burger's value");
    
	double bx = force->numeric(FLERR,arg[15]);
    double by = force->numeric(FLERR,arg[16]);
    double bz = force->numeric(FLERR,arg[17]);
	double A = force->numeric(FLERR,arg[18]);
    double B = force->numeric(FLERR,arg[19]);
    double C = force->numeric(FLERR,arg[20]);
	if ((A == 0) and (B == 0) and (C == 0)) error->all(FLERR,"Illegal slip plane");
	double r0 = force->numeric(FLERR,arg[21]);
	if (r0 <0) error->all(FLERR,"Illegal intial radius");
	//double r1 = force->numeric(FLERR,arg[22]);
	//if (r1 <0) error->all(FLERR,"Illegal intial radius");
	//double r2 = force->numeric(FLERR,arg[23]);
	//if (r2 <0) error->all(FLERR,"Illegal intial radius");
	double v = force->numeric(FLERR,arg[22]);
	int nopoint = force->numeric(FLERR,arg[23]);
	
    
	double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
	int flag;
	
	double X[3],Y[3],Z[3], X1[3],Y1[3],Z1[3], rini[3],xini[nopoint][3],m[3][3],xf[nopoint][3],r[nopoint+1],bv[3],A1,B1,C1,unitv[nopoint+1][3],a[nopoint][2],s[nopoint],E[nopoint],omega[nopoint],t[nopoint][6],fAB[nopoint][3],fBC[nopoint][3],fCA[nopoint][3], gAB[nopoint][3],gBC[nopoint][3],gCA[nopoint][3], u[3][3],signf;
	
	rini[0] = r0;
	//rini[1] = r1;
	//rini[2] = r2;
	
	
	
	// Burger's vector is X, normal vector to slip plane is Z, the 3rd unit vector is Y
	
	X[0] = bx / sqrt ( bx * bx + by * by + bz * bz);
	X[1] = by / sqrt ( bx * bx + by * by + bz * bz);
	X[2] = bz / sqrt ( bx * bx + by * by + bz * bz);
	
	Z[0] = A / sqrt ( A * A + B * B + C * C);
	Z[1] = B / sqrt ( A * A + B * B + C * C);
	Z[2] = C / sqrt ( A * A + B * B + C * C);
	
	Y[0] = Z[1] * X[2] - Z[2] * X[1];
	Y[1] = Z[2] * X[0] - Z[0] * X[2];
	Y[2] = Z[0] * X[1] - Z[1] * X[0];
	
	X1[0] = x1 / sqrt (x1 * x1 + x2 * x2 + x3 * x3);
	X1[1] = x2 / sqrt (x1 * x1 + x2 * x2 + x3 * x3);
	X1[2] = x3 / sqrt (x1 * x1 + x2 * x2 + x3 * x3);
	
	Y1[0] = y1 / sqrt (y1 * y1 + y2 * y2 + y3 * y3);
	Y1[1] = y2 / sqrt (y1 * y1 + y2 * y2 + y3 * y3);
	Y1[2] = y3 / sqrt (y1 * y1 + y2 * y2 + y3 * y3);
	
	Z1[0] = z1 / sqrt (z1 * z1 + z2 * z2 + z3 * z3);
	Z1[1] = z2 / sqrt (z1 * z1 + z2 * z2 + z3 * z3);
	Z1[2] = z3 / sqrt (z1 * z1 + z2 * z2 + z3 * z3);
	
	m[0][0] = X[0] * X1[0] + X[1] * X1[1] + X[2] * X1[2];
	m[0][1] = X[0] * Y1[0] + X[1] * Y1[1] + X[2] * Y1[2];
	m[0][2] = X[0] * Z1[0] + X[1] * Z1[1] + X[2] * Z1[2];
	m[1][0] = Y[0] * X1[0] + Y[1] * X1[1] + Y[2] * X1[2];
	m[1][1] = Y[0] * Y1[0] + Y[1] * Y1[1] + Y[2] * Y1[2];
	m[1][2] = Y[0] * Z1[0] + Y[1] * Z1[1] + Y[2] * Z1[2];
	m[2][0] = Z[0] * X1[0] + Z[1] * X1[1] + Z[2] * X1[2];
	m[2][1] = Z[0] * Y1[0] + Z[1] * Y1[1] + Z[2] * Y1[2];
	m[2][2] = Z[0] * Z1[0] + Z[1] * Z1[1] + Z[2] * Z1[2];
	

	
		
		bv[0] = (b * (X[0] * X1[0] + X[1] * X1[1] + X[2] * X1[2]));
		bv[1] = (b * (X[0] * Y1[0] + X[1] * Y1[1] + X[2] * Y1[2]));  
		bv[2] = (b * (X[0] * Z1[0] + X[1] * Z1[1] + X[2] * Z1[2])); 

		
		
		
		
		A1 = (Z[0] * X1[0] + Z[1] * X1[1] + Z[2] * X1[2]);
		B1 = (Z[0] * Y1[0] + Z[1] * Y1[1] + Z[2] * Y1[2]);
		C1 = (Z[0] * Z1[0] + Z[1] * Z1[1] + Z[2] * Z1[2]);
		
	
		
		//increment of angles depending on how many points are used to simulate the loop
		int n;
		double incre = 360/nopoint;
		double half = incre/2;
		
		for (i = 0; i < nlocal; i++) {
			if (mask[i] & groupbit) {
				
				int flag = 0;
				
				for (l = 0; l < 1; l++) {
					
					for (n = 0; n < nopoint; n++){
							xini[n][0] = rini[l] * cos ((half - n * incre) * M_PI / 180);
							xini[n][1] = rini[l] * sin ((half - n * incre) * M_PI / 180);
							xini[n][2] = 0;
						}
				for (j = 0; j < nopoint; j++) {	
				xf[j][0]= xini[j][0] * m[0][0] + xini[j][1] * m[1][0] + xini[j][2] * m[2][0] + x0;
				xf[j][1]= xini[j][0] * m[0][1] + xini[j][1] * m[1][1] + xini[j][2] * m[2][1] + y0;
				xf[j][2]= xini[j][0] * m[0][2] + xini[j][1] * m[1][2] + xini[j][2] * m[2][2] + z0;
			}
			
			 // check for overlap; if overlapp, displacement is set to zero
                        int counterr = 0;          
                        double testoverlap[nopoint];
                        for (j = 0 ; j < nopoint; j++){                  
                        testoverlap[j] = ( x[i][0] - xf[j][0] ) * ( x[i][0] - xf[j][0] ) + ( x[i][1] - xf[j][1] ) * ( x[i][1] - xf[j][1] ) + ( x[i][2] - xf[j][2] ) * ( x[i][2] - xf[j][2] );
			if (testoverlap[j] < 1e-4){
		 	counterr = counterr + 1;
			}
                        }

                        if ( counterr > 0) {        
                                u[0][0] = 0;     
                                u[1][0] = 0;
                                u[2][0] = 0;     
                        }                
                        else                            
                        {                
			
		
											
				r[0] =  sqrt(( x[i][0] - x0 ) * ( x[i][0] - x0 ) + ( x[i][1] - y0 ) * ( x[i][1] - y0 ) + ( x[i][2] - z0 ) * ( x[i][2] - z0 ) );
				for (j = 1; j < nopoint+1; j++) {
					r[j] = sqrt(( x[i][0] - xf[j-1][0] ) * ( x[i][0] - xf[j-1][0] ) + ( x[i][1] - xf[j-1][1] ) * ( x[i][1] - xf[j-1][1] ) + ( x[i][2] - xf[j-1][2] ) * ( x[i][2] - xf[j-1][2] ) );
				}	
			
				unitv[0][0] = ( x0 - x[i][0] ) / ( r[0] );
				unitv[0][1] = ( y0 - x[i][1] ) / ( r[0] );
				unitv[0][2] = ( z0 - x[i][2] ) / ( r[0] );
			
				for (j = 1; j < nopoint+1; j++) {
					unitv[j][0] = ( xf[j-1][0] - x[i][0] ) / ( r[j] );
					unitv[j][1] = ( xf[j-1][1] - x[i][1] ) / ( r[j] );
					unitv[j][2] = ( xf[j-1][2] - x[i][2] ) / ( r[j] );
					}
				
				for (j=1; j < nopoint+1; j++) {
					a[j-1][0] = acos(unitv[0][0]*unitv[j][0]+unitv[0][1]*unitv[j][1]+unitv[0][2]*unitv[j][2]);
					if (j < nopoint) {
						a[j-1][1] = acos(unitv[j][0]*unitv[j+1][0]+unitv[j][1]*unitv[j+1][1]+unitv[j][2]*unitv[j+1][2]);
					}
					else
					{
						a[j-1][1] = acos(unitv[j][0]*unitv[1][0]+unitv[j][1]*unitv[1][1]+unitv[j][2]*unitv[1][2]);
					}
				}
	
				for (j=0; j < nopoint; j++) {
					if (j < nopoint-1) {
						s[j] = (a[j][1] + a[j][0] + a[j+1][0]) / 2;
					}
					else
					{	
						s[j] = (a[j][1] + a[j][0] + a[0][0]) / 2;
					}
				}
		
				for (j = 0; j < nopoint; j++) {
					if (j < nopoint-1){
						double testangle = ( tan( s[j]/2 ) * tan( ( s[j] - a[j][1] ) / 2 ) * tan( ( s[j] - a[j][0] ) / 2 ) * tan( ( s[j] - a[j+1][0] ) / 2 ));
						if (testangle < 0){
							cout<<"testangle=";
							cout<<testangle<<endl;
							E[j] = 0;
						}
						else
						{
							E[j] = 4 * atan( sqrt ( tan( s[j]/2 ) * tan( ( s[j] - a[j][1] ) / 2 ) * tan( ( s[j] - a[j][0] ) / 2 ) * tan( ( s[j] - a[j+1][0] ) / 2 )));
						}
					}
					else
					{
						double testangle1 = ( tan( s[j]/2 ) * tan( ( s[j] - a[j][1] ) / 2 ) * tan( ( s[j] - a[j][0] ) / 2 ) * tan( ( s[j] - a[0][0] ) / 2 ));
						if (testangle1 < 0){
							cout<<"testangle1=";
							cout<<testangle1<<endl;
							E[j] = 0;
						}
						else
						{	
						E[j] = 4 * atan( sqrt ( tan( s[j]/2 ) * tan( ( s[j] - a[j][1] ) / 2 ) * tan( ( s[j] - a[j][0] ) / 2 ) * tan( ( s[j] - a[0][0] ) / 2 )));
						}
					}	
				}
		
				
					
						signf =  -unitv[0][0] * A1 - unitv[0][1] * B1 - unitv[0][2] * C1;
					
					
		
		
				for (j = 0; j < nopoint; j++) {
					omega[j] = -copysign(1.0, signf) * E[j];
				}
		
				for (j = 0; j < nopoint; j++) {
					t[j][0] = (xf[j][0] - x0) / sqrt((x0 - xf[j][0]) * (x0 - xf[j][0]) + (y0 - xf[j][1]) * (y0 - xf[j][1]) + (z0 - xf[j][2]) * (z0 - xf[j][2]));
					t[j][1] = (xf[j][1] - y0) / sqrt((x0 - xf[j][0]) * (x0 - xf[j][0]) + (y0 - xf[j][1]) * (y0 - xf[j][1]) + (z0 - xf[j][2]) * (z0 - xf[j][2]));
					t[j][2] = (xf[j][2] - z0) / sqrt((x0 - xf[j][0]) * (x0 - xf[j][0]) + (y0 - xf[j][1]) * (y0 - xf[j][1]) + (z0 - xf[j][2]) * (z0 - xf[j][2]));
					if (j < nopoint-1) {
						t[j][3] = (xf[j+1][0] - xf[j][0]) / sqrt((xf[j][0] - xf[j+1][0]) * (xf[j][0] - xf[j+1][0]) + (xf[j][1] - xf[j+1][1]) * (xf[j][1] - xf[j+1][1]) + (xf[j][2] - xf[j+1][2]) * (xf[j][2] - xf[j+1][2]));
						t[j][4] = (xf[j+1][1] - xf[j][1]) / sqrt((xf[j][0] - xf[j+1][0]) * (xf[j][0] - xf[j+1][0]) + (xf[j][1] - xf[j+1][1]) * (xf[j][1] - xf[j+1][1]) + (xf[j][2] - xf[j+1][2]) * (xf[j][2] - xf[j+1][2]));
						t[j][5] = (xf[j+1][2] - xf[j][2]) / sqrt((xf[j][0] - xf[j+1][0]) * (xf[j][0] - xf[j+1][0]) + (xf[j][1] - xf[j+1][1]) * (xf[j][1] - xf[j+1][1]) + (xf[j][2] - xf[j+1][2]) * (xf[j][2] - xf[j+1][2]));
					}
					else
					{
						t[j][3] = (xf[0][0] - xf[nopoint-1][0]) / sqrt((xf[nopoint-1][0] - xf[0][0]) * (xf[nopoint-1][0] - xf[0][0]) + (xf[nopoint-1][1] - xf[0][1]) * (xf[nopoint-1][1] - xf[0][1]) + (xf[nopoint-1][2] - xf[0][2]) * (xf[nopoint-1][2] - xf[0][2]));
						t[j][4] = (xf[0][1] - xf[nopoint-1][1]) / sqrt((xf[nopoint-1][0] - xf[0][0]) * (xf[nopoint-1][0] - xf[0][0]) + (xf[nopoint-1][1] - xf[0][1]) * (xf[nopoint-1][1] - xf[0][1]) + (xf[nopoint-1][2] - xf[0][2]) * (xf[nopoint-1][2] - xf[0][2]));
						t[j][5] = (xf[0][2] - xf[nopoint-1][2]) / sqrt((xf[nopoint-1][0] - xf[0][0]) * (xf[nopoint-1][0] - xf[0][0]) + (xf[nopoint-1][1] - xf[0][1]) * (xf[nopoint-1][1] - xf[0][1]) + (xf[nopoint-1][2] - xf[0][2]) * (xf[nopoint-1][2] - xf[0][2]));
					}
				}
								
					
				
				for (j = 0; j < nopoint; j++) {
			
				
				// check if any discontinuity in displacement function and output warning
					double test = fabs(unitv[0][0] * t[j][0] + unitv[0][1] * t[j][1] + unitv[0][2] * t[j][2]+ 1);
					if (test < 1e-14){
					cout<<"test=";
					cout<<test<<endl;
					}
					double lntest = (((unitv[j+1][0] * t[j][0] + unitv[j+1][1] * t[j][1] + unitv[j+1][2] * t[j][2] + 1) / (unitv[0][0] * t[j][0] + unitv[0][1] * t[j][1] + unitv[0][2] * t[j][2] + 1)) * ( r[j+1] / r[0]));
					if (lntest < 0){
					cout<<"lntest=";
					cout<<lntest<<endl;
					}
					
					
					if (fabs(unitv[0][0] * t[j][0] + unitv[0][1] * t[j][1] + unitv[0][2] * t[j][2] + 1) <1e-14) {
						error->warning(FLERR,"fAB term goes to infinity a");
						flag = 1;
					}
					else
					{	
					fAB[j][0] = (bv[1] * t[j][2] - bv[2] * t[j][1]) * log(((unitv[j+1][0] * t[j][0] + unitv[j+1][1] * t[j][1] + unitv[j+1][2] * t[j][2] + 1) / (unitv[0][0] * t[j][0] + unitv[0][1] * t[j][1] + unitv[0][2] * t[j][2] + 1)) * ( r[j+1] / r[0]));
					fAB[j][1] = (bv[2] * t[j][0] - bv[0] * t[j][2]) * log(((unitv[j+1][0] * t[j][0] + unitv[j+1][1] * t[j][1] + unitv[j+1][2] * t[j][2] + 1) / (unitv[0][0] * t[j][0] + unitv[0][1] * t[j][1] + unitv[0][2] * t[j][2] + 1)) * ( r[j+1] / r[0]));
					fAB[j][2] = (bv[0] * t[j][1] - bv[1] * t[j][0]) * log(((unitv[j+1][0] * t[j][0] + unitv[j+1][1] * t[j][1] + unitv[j+1][2] * t[j][2] + 1) / (unitv[0][0] * t[j][0] + unitv[0][1] * t[j][1] + unitv[0][2] * t[j][2] + 1)) * ( r[j+1] / r[0]));
					}
					if (j < nopoint-1) {
						
						double test1 = fabs(unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1);
						if (test1 < 1e-14){
							cout<<"test1=";
							cout<<test1<<endl;
							}
						
						double lntest1 = (((unitv[j+2][0] * t[j][3] + unitv[j+2][1] * t[j][4] + unitv[j+2][2] * t[j][5] + 1) / (unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1)) * ( r[j+2] / r[j+1]));
						if (lntest1 < 0){
							cout<<"lntest1=";
							cout<<lntest1<<endl;
							}
						
						if (fabs(unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1) <1e-14  ) {
							
							error->warning(FLERR,"fBC term goes to infinity at point x = ");
							flag = 1;
						
						}
						else
						{
							fBC[j][0] = (bv[1] * t[j][5] - bv[2] * t[j][4]) * log(((unitv[j+2][0] * t[j][3] + unitv[j+2][1] * t[j][4] + unitv[j+2][2] * t[j][5] + 1) / (unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1)) * ( r[j+2] / r[j+1]));
							fBC[j][1] = (bv[2] * t[j][3] - bv[0] * t[j][5]) * log(((unitv[j+2][0] * t[j][3] + unitv[j+2][1] * t[j][4] + unitv[j+2][2] * t[j][5] + 1) / (unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1)) * ( r[j+2] / r[j+1]));
							fBC[j][2] = (bv[0] * t[j][4] - bv[1] * t[j][3]) * log(((unitv[j+2][0] * t[j][3] + unitv[j+2][1] * t[j][4] + unitv[j+2][2] * t[j][5] + 1) / (unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1)) * ( r[j+2] / r[j+1]));
						}
						
						double test2 = fabs(unitv[j+2][0] * (-t[j+1][0]) + unitv[j+2][1] * (-t[j+1][1]) + unitv[j+2][2] * (-t[j+1][2]) + 1);
						if (test2 < 1e-14){
						cout<<"test2=";
						cout<<test2<<endl;
						}
						
						double lntest2 = (((unitv[0][0] * (-t[j+1][0]) + unitv[0][1] * (-t[j+1][1]) + unitv[0][2] * (-t[j+1][2]) + 1) / (unitv[j+2][0] * (-t[j+1][0]) + unitv[j+2][1] * (-t[j+1][1]) + unitv[j+2][2] * (-t[j+1][2]) + 1)) * ( r[0] / r[j+2]));
					if (lntest2 < 0){
					cout<<"lntest2=";
					cout<<lntest2<<endl;
					}
						
						
						if (fabs(unitv[j+2][0] * (-t[j+1][0]) + unitv[j+2][1] * (-t[j+1][1]) + unitv[j+2][2] * (-t[j+1][2]) + 1) <1e-14  ) {
							
							error->warning(FLERR,"fCA term goes to infinity at point x = ");
							flag = 1;
						}
						else
						{	
							fCA[j][0] = (bv[1] * (-t[j+1][2]) - bv[2] * (-t[j+1][1])) * log(((unitv[0][0] * (-t[j+1][0]) + unitv[0][1] * (-t[j+1][1]) + unitv[0][2] * (-t[j+1][2]) + 1) / (unitv[j+2][0] * (-t[j+1][0]) + unitv[j+2][1] * (-t[j+1][1]) + unitv[j+2][2] * (-t[j+1][2]) + 1)) * ( r[0] / r[j+2]));
							fCA[j][1] = (bv[2] * (-t[j+1][0]) - bv[0] * (-t[j+1][2])) * log(((unitv[0][0] * (-t[j+1][0]) + unitv[0][1] * (-t[j+1][1]) + unitv[0][2] * (-t[j+1][2]) + 1) / (unitv[j+2][0] * (-t[j+1][0]) + unitv[j+2][1] * (-t[j+1][1]) + unitv[j+2][2] * (-t[j+1][2]) + 1)) * ( r[0] / r[j+2]));
							fCA[j][2] = (bv[0] * (-t[j+1][1]) - bv[1] * (-t[j+1][0])) * log(((unitv[0][0] * (-t[j+1][0]) + unitv[0][1] * (-t[j+1][1]) + unitv[0][2] * (-t[j+1][2]) + 1) / (unitv[j+2][0] * (-t[j+1][0]) + unitv[j+2][1] * (-t[j+1][1]) + unitv[j+2][2] * (-t[j+1][2]) + 1)) * ( r[0] / r[j+2]));
						}
					}
					else
					{
						double test3 = fabs(unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1);
						if (test3 < 1e-14){
						cout<<"test3=";
						cout<<test3<<endl;
						}
						
						double lntest3 = (((unitv[1][0] * t[j][3] + unitv[1][1] * t[j][4] + unitv[1][2] * t[j][5] + 1) / (unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1)) * ( r[1] / r[j+1]));
					if (lntest3 < 0){
					cout<<"lntest3=";
					cout<<lntest3<<endl;
					}
						
						
						
						if (fabs(unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1) <1e-14  ) {
							
							error->warning(FLERR,"fBC term goes to infinity at point x = " );
							flag = 1;
						}
						else
						{
							fBC[j][0] = (bv[1] * t[j][5] - bv[2] * t[j][4]) * log(((unitv[1][0] * t[j][3] + unitv[1][1] * t[j][4] + unitv[1][2] * t[j][5] + 1) / (unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1)) * ( r[1] / r[j+1]));
							fBC[j][1] = (bv[2] * t[j][3] - bv[0] * t[j][5]) * log(((unitv[1][0] * t[j][3] + unitv[1][1] * t[j][4] + unitv[1][2] * t[j][5] + 1) / (unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1)) * ( r[1] / r[j+1]));
							fBC[j][2] = (bv[0] * t[j][4] - bv[1] * t[j][3]) * log(((unitv[1][0] * t[j][3] + unitv[1][1] * t[j][4] + unitv[1][2] * t[j][5] + 1) / (unitv[j+1][0] * t[j][3] + unitv[j+1][1] * t[j][4] + unitv[j+1][2] * t[j][5] + 1)) * ( r[1] / r[j+1]));
						}
						double test4 = fabs(unitv[1][0] * (-t[0][0]) + unitv[1][1] * (-t[0][1]) + unitv[1][2] * (-t[0][2]) + 1) ;
						if (test4 < 1e-14){
						cout<<"test4=";
						cout<<test4<<endl;
						}
						
						
						double lntest4 = (((unitv[0][0] * (-t[0][0]) + unitv[0][1] * (-t[0][1]) + unitv[0][2] * (-t[0][2]) + 1) / (unitv[1][0] * (-t[0][0]) + unitv[1][1] * (-t[0][1]) + unitv[1][2] * (-t[0][2]) + 1)) * ( r[0] / r[1]));
					if (lntest4 < 0){
					cout<<"lntest4=";
					cout<<lntest4<<endl;
					}
						
						
						if (fabs(unitv[1][0] * (-t[0][0]) + unitv[1][1] * (-t[0][1]) + unitv[1][2] * (-t[0][2]) + 1) <1e-14  ) {
							
							error->warning(FLERR,"fCA term goes to infinity at point x = " );
							flag = 1;
						}
						else
						{
							fCA[j][0] = (bv[1] * (-t[0][2]) - bv[2] * (-t[0][1])) * log(((unitv[0][0] * (-t[0][0]) + unitv[0][1] * (-t[0][1]) + unitv[0][2] * (-t[0][2]) + 1) / (unitv[1][0] * (-t[0][0]) + unitv[1][1] * (-t[0][1]) + unitv[1][2] * (-t[0][2]) + 1)) * ( r[0] / r[1]));
							fCA[j][1] = (bv[2] * (-t[0][0]) - bv[0] * (-t[0][2])) * log(((unitv[0][0] * (-t[0][0]) + unitv[0][1] * (-t[0][1]) + unitv[0][2] * (-t[0][2]) + 1) / (unitv[1][0] * (-t[0][0]) + unitv[1][1] * (-t[0][1]) + unitv[1][2] * (-t[0][2]) + 1)) * ( r[0] / r[1]));
							fCA[j][2] = (bv[0] * (-t[0][1]) - bv[1] * (-t[0][0])) * log(((unitv[0][0] * (-t[0][0]) + unitv[0][1] * (-t[0][1]) + unitv[0][2] * (-t[0][2]) + 1) / (unitv[1][0] * (-t[0][0]) + unitv[1][1] * (-t[0][1]) + unitv[1][2] * (-t[0][2]) + 1)) * ( r[0] / r[1]));
						}
					}
				}
			
				for (j = 0; j < nopoint; j++) {
					double test5 = fabs(1 + unitv[0][0] * unitv[j+1][0] + unitv[0][1] * unitv[j+1][1] + unitv[0][2] * unitv[j+1][2]);
					if (test5 < 1e-14){
					cout<<"test5=";
					cout<<test5<<endl;
					}
					if (fabs(1 + unitv[0][0] * unitv[j+1][0] + unitv[0][1] * unitv[j+1][1] + unitv[0][2] * unitv[j+1][2]) <1e-14  ) {
						
						error->warning(FLERR,"gAB term goes to infinity at point x = " );
						flag = 1;
					}
					else
					{
						gAB[j][0] = ((bv[0] * (unitv[0][1] * unitv[j+1][2] - unitv[0][2] * unitv[j+1][1] ) + bv[1] * (unitv[0][2] * unitv[j+1][0] - unitv[0][0] * unitv[j+1][2]) + bv[2] * (unitv[0][0] * unitv[j+1][1] - unitv[0][1] * unitv[j+1][0])) / (1 + unitv[0][0] * unitv[j+1][0] + unitv[0][1] * unitv[j+1][1] + unitv[0][2] * unitv[j+1][2])) * (unitv[0][0] + unitv[j+1][0]);
						gAB[j][1] = ((bv[0] * (unitv[0][1] * unitv[j+1][2] - unitv[0][2] * unitv[j+1][1] ) + bv[1] * (unitv[0][2] * unitv[j+1][0] - unitv[0][0] * unitv[j+1][2]) + bv[2] * (unitv[0][0] * unitv[j+1][1] - unitv[0][1] * unitv[j+1][0])) / (1 + unitv[0][0] * unitv[j+1][0] + unitv[0][1] * unitv[j+1][1] + unitv[0][2] * unitv[j+1][2])) * (unitv[0][1] + unitv[j+1][1]);
						gAB[j][2] = ((bv[0] * (unitv[0][1] * unitv[j+1][2] - unitv[0][2] * unitv[j+1][1] ) + bv[1] * (unitv[0][2] * unitv[j+1][0] - unitv[0][0] * unitv[j+1][2]) + bv[2] * (unitv[0][0] * unitv[j+1][1] - unitv[0][1] * unitv[j+1][0])) / (1 + unitv[0][0] * unitv[j+1][0] + unitv[0][1] * unitv[j+1][1] + unitv[0][2] * unitv[j+1][2])) * (unitv[0][2] + unitv[j+1][2]);
					}
					
					//wrong formula
					//gAB[j][0] = (bv[0] * (unitv[0][1] * unitv[j+1][2] - unitv[0][2] * unitv[j+1][1] ) + bv[1] * (unitv[0][2] * unitv[j+1][0] - unitv[0][0] * unitv[j+1][2]) + bv[2] * (unitv[0][0] * unitv[j+1][1] - unitv[0][1] * unitv[j+1][0])) * (1 - (unitv[0][0] * unitv[j+1][0] + unitv[0][1] * unitv[j+1][1] + unitv[0][2] * unitv[j+1][2])) * (unitv[0][0] + unitv[j+1][0]);
					//gAB[j][1] = (bv[0] * (unitv[0][1] * unitv[j+1][2] - unitv[0][2] * unitv[j+1][1] ) + bv[1] * (unitv[0][2] * unitv[j+1][0] - unitv[0][0] * unitv[j+1][2]) + bv[2] * (unitv[0][0] * unitv[j+1][1] - unitv[0][1] * unitv[j+1][0])) * (1 - (unitv[0][0] * unitv[j+1][0] + unitv[0][1] * unitv[j+1][1] + unitv[0][2] * unitv[j+1][2])) * (unitv[0][1] + unitv[j+1][1]);
					//gAB[j][2] = (bv[0] * (unitv[0][1] * unitv[j+1][2] - unitv[0][2] * unitv[j+1][1] ) + bv[1] * (unitv[0][2] * unitv[j+1][0] - unitv[0][0] * unitv[j+1][2]) + bv[2] * (unitv[0][0] * unitv[j+1][1] - unitv[0][1] * unitv[j+1][0])) * (1 - (unitv[0][0] * unitv[j+1][0] + unitv[0][1] * unitv[j+1][1] + unitv[0][2] * unitv[j+1][2])) * (unitv[0][2] + unitv[j+1][2]);
					if (j < nopoint-1) {
						double test6 = fabs(1 + unitv[j+1][0] * unitv[j+2][0] + unitv[j+1][1] * unitv[j+2][1] + unitv[j+1][2] * unitv[j+2][2]);
						if (test6 < 1e-14){
						cout<<"test6=";
						cout<<test6<<endl;
						}
						if (fabs(1 + unitv[j+1][0] * unitv[j+2][0] + unitv[j+1][1] * unitv[j+2][1] + unitv[j+1][2] * unitv[j+2][2]) <1e-14  ) {
							
							error->warning(FLERR,"gBC term goes to infinity at point x = ");
							flag = 1;
						}
						else
						{
							gBC[j][0] = ((bv[0] * (unitv[j+1][1] * unitv[j+2][2] - unitv[j+1][2] * unitv[j+2][1] ) + bv[1] * (unitv[j+1][2] * unitv[j+2][0] - unitv[j+1][0] * unitv[j+2][2]) + bv[2] * (unitv[j+1][0] * unitv[j+2][1] - unitv[j+1][1] * unitv[j+2][0])) / (1 + unitv[j+1][0] * unitv[j+2][0] + unitv[j+1][1] * unitv[j+2][1] + unitv[j+1][2] * unitv[j+2][2])) * (unitv[j+1][0] + unitv[j+2][0]);
							gBC[j][1] = ((bv[0] * (unitv[j+1][1] * unitv[j+2][2] - unitv[j+1][2] * unitv[j+2][1] ) + bv[1] * (unitv[j+1][2] * unitv[j+2][0] - unitv[j+1][0] * unitv[j+2][2]) + bv[2] * (unitv[j+1][0] * unitv[j+2][1] - unitv[j+1][1] * unitv[j+2][0])) / (1 + unitv[j+1][0] * unitv[j+2][0] + unitv[j+1][1] * unitv[j+2][1] + unitv[j+1][2] * unitv[j+2][2])) * (unitv[j+1][1] + unitv[j+2][1]);
							gBC[j][2] = ((bv[0] * (unitv[j+1][1] * unitv[j+2][2] - unitv[j+1][2] * unitv[j+2][1] ) + bv[1] * (unitv[j+1][2] * unitv[j+2][0] - unitv[j+1][0] * unitv[j+2][2]) + bv[2] * (unitv[j+1][0] * unitv[j+2][1] - unitv[j+1][1] * unitv[j+2][0])) / (1 + unitv[j+1][0] * unitv[j+2][0] + unitv[j+1][1] * unitv[j+2][1] + unitv[j+1][2] * unitv[j+2][2])) * (unitv[j+1][2] + unitv[j+2][2]);
						}
						
						
						double test7 = fabs(1 + unitv[j+2][0] * unitv[0][0] + unitv[j+2][1] * unitv[0][1] + unitv[j+2][2] * unitv[0][2]);
						if (test7 < 1e-14){
						cout<<"test7=";
						cout<<test7<<endl;
						}
						if (fabs(1 + unitv[j+2][0] * unitv[0][0] + unitv[j+2][1] * unitv[0][1] + unitv[j+2][2] * unitv[0][2]) <1e-14  ) {
							
							error->warning(FLERR,"gCA term goes to infinity at point x = " );
							flag = 1;
						}
						else
						{
							gCA[j][0] = ((bv[0] * (unitv[j+2][1] * unitv[0][2] - unitv[j+2][2] * unitv[0][1] ) + bv[1] * (unitv[j+2][2] * unitv[0][0] - unitv[j+2][0] * unitv[0][2]) + bv[2] * (unitv[j+2][0] * unitv[0][1] - unitv[j+2][1] * unitv[0][0])) / (1 + unitv[j+2][0] * unitv[0][0] + unitv[j+2][1] * unitv[0][1] + unitv[j+2][2] * unitv[0][2])) * (unitv[j+2][0] + unitv[0][0]);
							gCA[j][1] = ((bv[0] * (unitv[j+2][1] * unitv[0][2] - unitv[j+2][2] * unitv[0][1] ) + bv[1] * (unitv[j+2][2] * unitv[0][0] - unitv[j+2][0] * unitv[0][2]) + bv[2] * (unitv[j+2][0] * unitv[0][1] - unitv[j+2][1] * unitv[0][0])) / (1 + unitv[j+2][0] * unitv[0][0] + unitv[j+2][1] * unitv[0][1] + unitv[j+2][2] * unitv[0][2])) * (unitv[j+2][1] + unitv[0][1]);
							gCA[j][2] = ((bv[0] * (unitv[j+2][1] * unitv[0][2] - unitv[j+2][2] * unitv[0][1] ) + bv[1] * (unitv[j+2][2] * unitv[0][0] - unitv[j+2][0] * unitv[0][2]) + bv[2] * (unitv[j+2][0] * unitv[0][1] - unitv[j+2][1] * unitv[0][0])) / (1 + unitv[j+2][0] * unitv[0][0] + unitv[j+2][1] * unitv[0][1] + unitv[j+2][2] * unitv[0][2])) * (unitv[j+2][2] + unitv[0][2]);
						}
						//wrong formula
						//gBC[j][0] = (bv[0] * (unitv[j+1][1] * unitv[j+2][2] - unitv[j+1][2] * unitv[j+2][1] ) + bv[1] * (unitv[j+1][2] * unitv[j+2][0] - unitv[j+1][0] * unitv[j+2][2]) + bv[2] * (unitv[j+1][0] * unitv[j+2][1] - unitv[j+1][1] * unitv[j+2][0])) * (1 - (unitv[j+1][0] * unitv[j+2][0] + unitv[j+1][1] * unitv[j+2][1] + unitv[j+1][2] * unitv[j+2][2])) * (unitv[j+1][0] + unitv[j+2][0]);
						//gBC[j][1] = (bv[0] * (unitv[j+1][1] * unitv[j+2][2] - unitv[j+1][2] * unitv[j+2][1] ) + bv[1] * (unitv[j+1][2] * unitv[j+2][0] - unitv[j+1][0] * unitv[j+2][2]) + bv[2] * (unitv[j+1][0] * unitv[j+2][1] - unitv[j+1][1] * unitv[j+2][0])) * (1 - (unitv[j+1][0] * unitv[j+2][0] + unitv[j+1][1] * unitv[j+2][1] + unitv[j+1][2] * unitv[j+2][2])) * (unitv[j+1][1] + unitv[j+2][1]);
						//gBC[j][2] = (bv[0] * (unitv[j+1][1] * unitv[j+2][2] - unitv[j+1][2] * unitv[j+2][1] ) + bv[1] * (unitv[j+1][2] * unitv[j+2][0] - unitv[j+1][0] * unitv[j+2][2]) + bv[2] * (unitv[j+1][0] * unitv[j+2][1] - unitv[j+1][1] * unitv[j+2][0])) * (1 - (unitv[j+1][0] * unitv[j+2][0] + unitv[j+1][1] * unitv[j+2][1] + unitv[j+1][2] * unitv[j+2][2])) * (unitv[j+1][2] + unitv[j+2][2]);
					
						//gCA[j][0] = (bv[0] * (unitv[j+2][1] * unitv[0][2] - unitv[j+2][2] * unitv[0][1] ) + bv[1] * (unitv[j+2][2] * unitv[0][0] - unitv[j+2][0] * unitv[0][2]) + bv[2] * (unitv[j+2][0] * unitv[0][1] - unitv[j+2][1] * unitv[0][0])) * (1 - (unitv[j+2][0] * unitv[0][0] + unitv[j+2][1] * unitv[0][1] + unitv[j+2][2] * unitv[0][2])) * (unitv[j+2][0] + unitv[0][0]);
						//gCA[j][1] = (bv[0] * (unitv[j+2][1] * unitv[0][2] - unitv[j+2][2] * unitv[0][1] ) + bv[1] * (unitv[j+2][2] * unitv[0][0] - unitv[j+2][0] * unitv[0][2]) + bv[2] * (unitv[j+2][0] * unitv[0][1] - unitv[j+2][1] * unitv[0][0])) * (1 - (unitv[j+2][0] * unitv[0][0] + unitv[j+2][1] * unitv[0][1] + unitv[j+2][2] * unitv[0][2])) * (unitv[j+2][1] + unitv[0][1]);
						//gCA[j][2] = (bv[0] * (unitv[j+2][1] * unitv[0][2] - unitv[j+2][2] * unitv[0][1] ) + bv[1] * (unitv[j+2][2] * unitv[0][0] - unitv[j+2][0] * unitv[0][2]) + bv[2] * (unitv[j+2][0] * unitv[0][1] - unitv[j+2][1] * unitv[0][0])) * (1 - (unitv[j+2][0] * unitv[0][0] + unitv[j+2][1] * unitv[0][1] + unitv[j+2][2] * unitv[0][2])) * (unitv[j+2][2] + unitv[0][2]);
					
					}
					else
					{
						double test8 = fabs(1 + unitv[j+1][0] * unitv[1][0] + unitv[j+1][1] * unitv[1][1] + unitv[j+1][2] * unitv[1][2]);
						if (test8 < 1e-14){
						cout<<"test8=";
						cout<<test8<<endl;
						}
						if (fabs(1 + unitv[j+1][0] * unitv[1][0] + unitv[j+1][1] * unitv[1][1] + unitv[j+1][2] * unitv[1][2]) <1e-14  ) {
							
							error->warning(FLERR,"gBC term goes to infinity at point x = ");
							flag = 1;
						}
						else
						{
							gBC[j][0] = ((bv[0] * (unitv[j+1][1] * unitv[1][2] - unitv[j+1][2] * unitv[1][1] ) + bv[1] * (unitv[j+1][2] * unitv[1][0] - unitv[j+1][0] * unitv[1][2]) + bv[2] * (unitv[j+1][0] * unitv[1][1] - unitv[j+1][1] * unitv[1][0])) / (1 + unitv[j+1][0] * unitv[1][0] + unitv[j+1][1] * unitv[1][1] + unitv[j+1][2] * unitv[1][2])) * (unitv[j+1][0] + unitv[1][0]);
							gBC[j][1] = ((bv[0] * (unitv[j+1][1] * unitv[1][2] - unitv[j+1][2] * unitv[1][1] ) + bv[1] * (unitv[j+1][2] * unitv[1][0] - unitv[j+1][0] * unitv[1][2]) + bv[2] * (unitv[j+1][0] * unitv[1][1] - unitv[j+1][1] * unitv[1][0])) / (1 + unitv[j+1][0] * unitv[1][0] + unitv[j+1][1] * unitv[1][1] + unitv[j+1][2] * unitv[1][2])) * (unitv[j+1][1] + unitv[1][1]);
							gBC[j][2] = ((bv[0] * (unitv[j+1][1] * unitv[1][2] - unitv[j+1][2] * unitv[1][1] ) + bv[1] * (unitv[j+1][2] * unitv[1][0] - unitv[j+1][0] * unitv[1][2]) + bv[2] * (unitv[j+1][0] * unitv[1][1] - unitv[j+1][1] * unitv[1][0])) / (1 + unitv[j+1][0] * unitv[1][0] + unitv[j+1][1] * unitv[1][1] + unitv[j+1][2] * unitv[1][2])) * (unitv[j+1][2] + unitv[1][2]);
						}
						
						double test9 = fabs(1 + unitv[1][0] * unitv[0][0] + unitv[1][1] * unitv[0][1] + unitv[1][2] * unitv[0][2]);
						if (test9 < 1e-14){
						cout<<"test9=";
						cout<<test9<<endl;
						} 
						
						if (fabs(1 + unitv[1][0] * unitv[0][0] + unitv[1][1] * unitv[0][1] + unitv[1][2] * unitv[0][2]) <1e-14  ) {
							
							error->warning(FLERR,"gCA term goes to infinity at point x = ");
							flag = 1;
						}
						else
						{
							gCA[j][0] = ((bv[0] * (unitv[1][1] * unitv[0][2] - unitv[1][2] * unitv[0][1] ) + bv[1] * (unitv[1][2] * unitv[0][0] - unitv[1][0] * unitv[0][2]) + bv[2] * (unitv[1][0] * unitv[0][1] - unitv[1][1] * unitv[0][0])) / (1 + unitv[1][0] * unitv[0][0] + unitv[1][1] * unitv[0][1] + unitv[1][2] * unitv[0][2])) * (unitv[1][0] + unitv[0][0]);
							gCA[j][1] = ((bv[0] * (unitv[1][1] * unitv[0][2] - unitv[1][2] * unitv[0][1] ) + bv[1] * (unitv[1][2] * unitv[0][0] - unitv[1][0] * unitv[0][2]) + bv[2] * (unitv[1][0] * unitv[0][1] - unitv[1][1] * unitv[0][0])) / (1 + unitv[1][0] * unitv[0][0] + unitv[1][1] * unitv[0][1] + unitv[1][2] * unitv[0][2])) * (unitv[1][1] + unitv[0][1]);
							gCA[j][2] = ((bv[0] * (unitv[1][1] * unitv[0][2] - unitv[1][2] * unitv[0][1] ) + bv[1] * (unitv[1][2] * unitv[0][0] - unitv[1][0] * unitv[0][2]) + bv[2] * (unitv[1][0] * unitv[0][1] - unitv[1][1] * unitv[0][0])) / (1 + unitv[1][0] * unitv[0][0] + unitv[1][1] * unitv[0][1] + unitv[1][2] * unitv[0][2])) * (unitv[1][2] + unitv[0][2]);
						}
						//wrong formula
						//gBC[j][0] = (bv[0] * (unitv[j+1][1] * unitv[1][2] - unitv[j+1][2] * unitv[1][1] ) + bv[1] * (unitv[j+1][2] * unitv[1][0] - unitv[j+1][0] * unitv[1][2]) + bv[2] * (unitv[j+1][0] * unitv[1][1] - unitv[j+1][1] * unitv[1][0])) * (1 - (unitv[j+1][0] * unitv[1][0] + unitv[j+1][1] * unitv[1][1] + unitv[j+1][2] * unitv[1][2])) * (unitv[j+1][0] + unitv[1][0]);
						//gBC[j][1] = (bv[0] * (unitv[j+1][1] * unitv[1][2] - unitv[j+1][2] * unitv[1][1] ) + bv[1] * (unitv[j+1][2] * unitv[1][0] - unitv[j+1][0] * unitv[1][2]) + bv[2] * (unitv[j+1][0] * unitv[1][1] - unitv[j+1][1] * unitv[1][0])) * (1 - (unitv[j+1][0] * unitv[1][0] + unitv[j+1][1] * unitv[1][1] + unitv[j+1][2] * unitv[1][2])) * (unitv[j+1][1] + unitv[1][1]);
						//gBC[j][2] = (bv[0] * (unitv[j+1][1] * unitv[1][2] - unitv[j+1][2] * unitv[1][1] ) + bv[1] * (unitv[j+1][2] * unitv[1][0] - unitv[j+1][0] * unitv[1][2]) + bv[2] * (unitv[j+1][0] * unitv[1][1] - unitv[j+1][1] * unitv[1][0])) * (1 - (unitv[j+1][0] * unitv[1][0] + unitv[j+1][1] * unitv[1][1] + unitv[j+1][2] * unitv[1][2])) * (unitv[j+1][2] + unitv[1][2]);
				
						//gCA[j][0] = (bv[0] * (unitv[1][1] * unitv[0][2] - unitv[1][2] * unitv[0][1] ) + bv[1] * (unitv[1][2] * unitv[0][0] - unitv[1][0] * unitv[0][2]) + bv[2] * (unitv[1][0] * unitv[0][1] - unitv[1][1] * unitv[0][0])) * (1 - (unitv[1][0] * unitv[0][0] + unitv[1][1] * unitv[0][1] + unitv[1][2] * unitv[0][2])) * (unitv[1][0] + unitv[0][0]);
						//gCA[j][1] = (bv[0] * (unitv[1][1] * unitv[0][2] - unitv[1][2] * unitv[0][1] ) + bv[1] * (unitv[1][2] * unitv[0][0] - unitv[1][0] * unitv[0][2]) + bv[2] * (unitv[1][0] * unitv[0][1] - unitv[1][1] * unitv[0][0])) * (1 - (unitv[1][0] * unitv[0][0] + unitv[1][1] * unitv[0][1] + unitv[1][2] * unitv[0][2])) * (unitv[1][1] + unitv[0][1]);
						//gCA[j][2] = (bv[0] * (unitv[1][1] * unitv[0][2] - unitv[1][2] * unitv[0][1] ) + bv[1] * (unitv[1][2] * unitv[0][0] - unitv[1][0] * unitv[0][2]) + bv[2] * (unitv[1][0] * unitv[0][1] - unitv[1][1] * unitv[0][0])) * (1 - (unitv[1][0] * unitv[0][0] + unitv[1][1] * unitv[0][1] + unitv[1][2] * unitv[0][2])) * (unitv[1][2] + unitv[0][2]);
					}
					}

				for (j = 0; j < 3; j++) {
					u[j][l] = 0;
				}
				
				for (j = 0; j < 3; j++) {
					for (k = 0; k < nopoint; k++) {
						if (flag == 1) {
							u[j][l] = 0;
						}
						else
						{	
							u[j][l]= u[j][l] - (bv[j] * omega[k] / (4 * MY_PI)) - (((1 - 2 * v) / (8 * MY_PI * (1 - v))) * ( fAB[k][j] + fBC[k][j] + fCA[k][j])) + ((1 / (8 * MY_PI * (1 - v))) * ( gAB[k][j] + gBC[k][j] + gCA[k][j]));
						}
					}
				}
				}
				
				if (fabs(u[0][0]) > 10 ) {
				
					cout<<"error happens at x=";
					cout<<x[i][0]<<endl;
					cout<<"error happens at y=";
					cout<<x[i][1]<<endl;
					cout<<"error happens at z=";
					cout<<x[i][2]<<endl;
				}
			//	x[i][0] += u[0][0] + u[0][1] + u[0][2];
			//	x[i][1] += u[1][0] + u[1][1] + u[1][2];
			//	x[i][2] += u[2][0] + u[2][1] + u[2][2];
			}			
				x[i][0] += u[0][0] ;
				x[i][1] += u[1][0] ;
				x[i][2] += u[2][0] ;

	}
	}
  }
  // rotate atoms bv[1] right-hand rule bv[1] theta around R
  // P = point = vector = point of rotation
  // R = vector = axis of rotation
  // R0 = runit = unit vector for R
  // D = X - P = vector from P to X
  // C = (D dot R0) R0 = projection of atom coord onto R line
  // A = D - C = vector from R line to X
  // B = R0 cross A = vector perp to A in plane of rotation
  // A,B define plane of circular rotation around R line
  // X = P + C + A cos(theta) + B sin(theta)

  if (style == ROTATE) {
    double axis[3],point[3];
    double a[3],b[3],c[3],d[3],disp[3],runit[3];
    
    int dim = domain->dimension;
    point[0] = xscale*force->numeric(FLERR,arg[2]);
    point[1] = yscale*force->numeric(FLERR,arg[3]);
    point[2] = zscale*force->numeric(FLERR,arg[4]);
    axis[0] = force->numeric(FLERR,arg[5]);
    axis[1] = force->numeric(FLERR,arg[6]);
    axis[2] = force->numeric(FLERR,arg[7]);
    double theta = force->numeric(FLERR,arg[8]);
    if (dim == 2 && (axis[0] != 0.0 || axis[1] != 0.0))
      error->all(FLERR,"Invalid displace_atoms rotate axis for 2d");

    double len = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (len == 0.0)
      error->all(FLERR,"Zero length rotation vector with displace_atoms");
    runit[0] = axis[0]/len;
    runit[1] = axis[1]/len;
    runit[2] = axis[2]/len;

    double sine = sin(MY_PI*theta/180.0);
    double cosine = cos(MY_PI*theta/180.0);
    double ddotr;

    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        d[0] = x[i][0] - point[0];
        d[1] = x[i][1] - point[1];
        d[2] = x[i][2] - point[2];
        ddotr = d[0]*runit[0] + d[1]*runit[1] + d[2]*runit[2];
        c[0] = ddotr*runit[0];
        c[1] = ddotr*runit[1];
        c[2] = ddotr*runit[2];
        a[0] = d[0] - c[0];
        a[1] = d[1] - c[1];
        a[2] = d[2] - c[2];
        b[0] = runit[1]*a[2] - runit[2]*a[1];
        b[1] = runit[2]*a[0] - runit[0]*a[2];
        b[2] = runit[0]*a[1] - runit[1]*a[0];
        disp[0] = a[0]*cosine  + b[0]*sine;
        disp[1] = a[1]*cosine  + b[1]*sine;
        disp[2] = a[2]*cosine  + b[2]*sine;
        x[i][0] = point[0] + c[0] + disp[0];
        x[i][1] = point[1] + c[1] + disp[1];
        if (dim == 3) x[i][2] = point[2] + c[2] + disp[2];
      }
    }
  }

  // generate single edge dislocations 


  if (style == SEDGE) {
	
    double x0 = xscale*force->numeric(FLERR,arg[2]);
    double y0 = yscale*force->numeric(FLERR,arg[3]);
	double b = force->numeric(FLERR,arg[4]);
    //if (b <= 0) error->all(FLERR,"Illegal Burger's value");
    double v = force->numeric(FLERR,arg[5]);
	
	double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

	double	xn,yn;
	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
		xn = x[i][0] - x0;
		yn = x[i][1] - y0;
		x[i][0] += b /(2 * MY_PI) * ( atan2(yn,xn) + ((xn * yn) / ( 2 * (1 - v) * (xn * xn + yn * yn))));  
		x[i][1] += - (b /(2 * MY_PI)) * (((1 - 2 * v) / ( 4 * (1 - v))) * log(xn * xn + yn * yn) + (xn * xn - yn * yn) / (4 * (1 - v) * (xn * xn + yn * yn)));
	}	
  }
  }
  // generate single screw dislocations


  if (style == SSCREW) {
	
    double x0 = xscale*force->numeric(FLERR,arg[2]);
    double y0 = yscale*force->numeric(FLERR,arg[3]);
	double b = force->numeric(FLERR,arg[4]);
    //if (b <= 0) error->all(FLERR,"Illegal Burger's value");
    
	
	double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

	double	xn,yn;
	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
		xn = x[i][0] - x0;
		yn = x[i][1] - y0;
		x[i][2] += b /(2 * MY_PI) * (atan2(yn,xn));  
		}	
  }
  }

// Compute Loop Size
  if (style == LOOPSIZE) {
  
	double x1 = force->numeric(FLERR,arg[2]);
    double x2 = force->numeric(FLERR,arg[3]);
    double x3 = force->numeric(FLERR,arg[4]);
	double y1 = force->numeric(FLERR,arg[5]);
    double y2 = force->numeric(FLERR,arg[6]);
    double y3 = force->numeric(FLERR,arg[7]);
	double z1 = force->numeric(FLERR,arg[8]);
    double z2 = force->numeric(FLERR,arg[9]);
    double z3 = force->numeric(FLERR,arg[10]);
	
    
	double bx = force->numeric(FLERR,arg[11]);
    double by = force->numeric(FLERR,arg[12]);
    double bz = force->numeric(FLERR,arg[13]);
	double A = force->numeric(FLERR,arg[14]);
    double B = force->numeric(FLERR,arg[15]);
    double C = force->numeric(FLERR,arg[16]);
	
	int	noave = force->numeric(FLERR,arg[17]);
	
	double **x = atom->x;
	int *mask = atom->mask;
    int nlocal = atom->nlocal;
	tagint *tag = atom->tag;
	int my_rank;
	
	double	X[3],Y[3],Z[3],X1[3],Y1[3],Z1[3],m[3][3];
	
	X[0] = bx / sqrt ( bx * bx + by * by + bz * bz);
	X[1] = by / sqrt ( bx * bx + by * by + bz * bz);
	X[2] = bz / sqrt ( bx * bx + by * by + bz * bz);
	
	Z[0] = A / sqrt ( A * A + B * B + C * C);
	Z[1] = B / sqrt ( A * A + B * B + C * C);
	Z[2] = C / sqrt ( A * A + B * B + C * C);
	
	Y[0] = Z[1] * X[2] - Z[2] * X[1];
	Y[1] = Z[2] * X[0] - Z[0] * X[2];
	Y[2] = Z[0] * X[1] - Z[1] * X[0];
	
	X1[0] = x1 / sqrt (x1 * x1 + x2 * x2 + x3 * x3);
	X1[1] = x2 / sqrt (x1 * x1 + x2 * x2 + x3 * x3);
	X1[2] = x3 / sqrt (x1 * x1 + x2 * x2 + x3 * x3);
	
	Y1[0] = y1 / sqrt (y1 * y1 + y2 * y2 + y3 * y3);
	Y1[1] = y2 / sqrt (y1 * y1 + y2 * y2 + y3 * y3);
	Y1[2] = y3 / sqrt (y1 * y1 + y2 * y2 + y3 * y3);
	
	Z1[0] = z1 / sqrt (z1 * z1 + z2 * z2 + z3 * z3);
	Z1[1] = z2 / sqrt (z1 * z1 + z2 * z2 + z3 * z3);
	Z1[2] = z3 / sqrt (z1 * z1 + z2 * z2 + z3 * z3);
	
	m[0][0] = X1[0] * X[0] + X1[1] * X[1] + X1[2] * X[2];
	m[0][1] = X1[0] * Y[0] + X1[1] * Y[1] + X1[2] * Y[2];
	m[0][2] = X1[0] * Z[0] + X1[1] * Z[1] + X1[2] * Z[2];
	
	m[1][0] = Y1[0] * X[0] + Y1[1] * X[1] + Y1[2] * X[2];
	m[1][1] = Y1[0] * Y[0] + Y1[1] * Y[1] + Y1[2] * Y[2];
	m[1][2] = Y1[0] * Z[0] + Y1[1] * Z[1] + Y1[2] * Z[2];
	
	m[2][0] = Z1[0] * X[0] + Z1[1] * X[1] + Z1[2] * X[2];
	m[2][1] = Z1[0] * Y[0] + Z1[1] * Y[1] + Z1[2] * Y[2];
	m[2][2] = Z1[0] * Z[0] + Z1[1] * Z[1] + Z1[2] * Z[2];
	
		
//	MPI_Init(&narg, &arg);
	
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	
	
		int number_of_lines = 0;
		ifstream fin("dump.disp");
		string line;
	if (my_rank == 0)
	{
		//count number of lines
		while (getline(fin,line))
		{
			number_of_lines = number_of_lines + 1;
		}
		
		cout<<endl<<"number_of_lines="<<number_of_lines;
		fin.clear( );
		fin.seekg( 0, std::ios::beg );
		
		//count number of atoms
		int noa;
		int start = 4;
		int end = 4;
		unsigned int count = 0;
		
		while (getline(fin,line))
		{
			count++;
			if (count > end) { break; }    // done
			if (count < start)  { continue; } // too early
			
			istringstream iss(line);
		
			iss >> noa;
		}
		cout<<endl<<"noa="<<noa;
		
		
		fin.clear( );
		fin.seekg( 0, std::ios::beg );
		
		//compute number of images in the dump file 
		int number_of_images = floor(number_of_lines/noa);
		
		
		int index[number_of_images],temp,temp1,k;
		double tempvectorx[noa],tempvectory[noa],tempvectorz[noa],inanglex[noa], inangley[noa],anglex[number_of_images], angley[number_of_images],tempmaxx[number_of_images], tempmaxy[number_of_images],tempmaxz[number_of_images], tempmaxx1[number_of_images], tempmaxy1[number_of_images],tempmaxz1[number_of_images],da[number_of_images],db[number_of_images],a[number_of_images],b[number_of_images],a1[number_of_images],b1[number_of_images],posx[noa],posy[noa],posz[noa],posx1[noa],posy1[noa],posz1[noa],posx2[noa],posy2[noa],posz2[noa],centro[noa],tempx,tempy,tempz,tempcentro;
		
		//1st loop to read all coordinate, transform to local coordinate and makes copy to sort later
		
		for (k = 1; k<number_of_images+1; k++){
				
				start = noa*(k-1)+9*k+1;
				end = (noa+9)*k;
				count = 0;
			
				while (getline(fin,line))
				{
			
					count++;
					if (count > end) { break; }    // done
					if (count < start)  { continue; } // too early
	
					istringstream iss(line);
		
					iss >> temp;
					iss >> temp1;	
					iss >> tempx;
					iss >> tempy;
					iss >> tempz;
					iss >> tempcentro;
		
			
					int j = temp - 1;
					posx[j] = tempx;
					posy[j] = tempy;
					posz[j] = tempz;
					centro[j]= tempcentro;
			
				}
					
				for (j = 0; j<noa; j++){
					tempx=posx[j];
					tempy=posy[j];
					tempz=posz[j];
					posx[j] = fabs(tempx*m[0][0]+tempy*m[1][0]+tempz*m[2][0]);
					posy[j] = fabs(tempx*m[0][1]+tempy*m[1][1]+tempz*m[2][1]);
					posz[j] = fabs(tempx*m[0][2]+tempy*m[1][2]+tempz*m[2][2]);
					posx1[j] = posx[j];
					posy1[j] = posy[j];
					posz1[j] = posz[j];
					posx2[j] = posx[j];
					posy2[j] = posy[j];
					posz2[j] = posz[j];
				}
		
				//find the maximum x and y and ensure any atom not in the loop is excluded, max x is leading partial position of edge, max y is leading partial of screw
				double maxx[k],maxy[k];
				//const int N = sizeof(posx1) / sizeof(double);
				
		
				double diffx = 10;
				double diffy = 10;
				int count1 = 0;
				
				while (diffx > 5) 
				{
					
					for (j = 0; j<noa; j++){
						if (count1>0)
						{
							posx1[j] = tempvectorx[j];
							posy1[j] = tempvectory[j];
							posz1[j] = tempvectorz[j];	
						}
						if ((posz[j] > 2) || (centro[j]<5))
						{
							posx1[j] = 0;
						}	
					}
					
					// set the maximum values that away from the loop 0
					
					if ((count1 > 1) && (diffx>5)) 
					{
						posx1[index[k]] = 0;
					}
					
										
					maxx[k] = *max_element(posx1,posx1 + noa);
					index[k]= distance(posx1, max_element(posx1, posx1 + noa));
					
					tempmaxx[k] = posx[index[k]];
					tempmaxy[k] = posy[index[k]];
					tempmaxz[k] = posz[index[k]];
					
					for (j = 0; j<noa; j++){
						tempvectorx[j] = posx1[j];
						tempvectory[j] = posy1[j];
						tempvectorz[j] = posz1[j];
					}
					sort(posx1,posx1+noa);
					double sumx = 0;	
			
					for (j = noa-1; j > noa-1-noave; j--){
						sumx += posx1[j]; 
					}
					cout<<endl<<"";
					a[k] = sumx/noave;
					diffx= fabs(a[k]-maxx[k]);
					count1 = count1+1;
				
				}
				
				count1 = 0;
				
				while (diffy > 5) 
				{
					for (j = 0; j<noa; j++){
						if (count1>0)
						{
							posx1[j] = tempvectorx[j];
							posy1[j] = tempvectory[j];
							posz1[j] = tempvectorz[j];	
						}
						
						if ((posz[j] > 2) || (centro[j]<5)) 
						{
							posy1[j] = 0;
						}
					}
					
					// set the maximum values that away from the loop 0
					
					if ((count1 > 1) && (diffy>5)) 
					{
						posy1[index[k]] = 0;
					}
					
					maxy[k] = *max_element(posy1,posy1+noa);
					index[k] = distance(posy1, max_element(posy1, posy1 + noa));
					tempmaxx1[k] = posx[index[k]];
					tempmaxy1[k] = posy[index[k]];
					tempmaxz1[k] = posz[index[k]];
					
					for (j = 0; j<noa; j++){
						tempvectorx[j] = posx1[j];
						tempvectory[j] = posy1[j];
						tempvectorz[j] = posz1[j];
					}
			
					sort(posy1,posy1+noa);
					double sumy = 0;	
			
					for (j = noa-1; j > noa-1-noave; j--){
						sumy += posy1[j]; 
					}
				
					b[k] = sumy/noave;
					diffy= fabs(b[k]-maxy[k]);
					count1 = count1+1;
					
				}
			
				
				//based on the defined arc length, determine the angle covering the edge/screw component --> find the minimum in that range to find the position of trailing (assumption is symmetric over 0,0,0)
				double arclength = 15.0;
				anglex[k] = arclength / tempmaxx[k];
				angley[k] = arclength / tempmaxy1[k];
				
				cout<<endl<<anglex[k]<<","<<angley[k];
				for (j = 0; j<noa; j++){
					
					inanglex [j] = fabs(atan2(posy[j],posx[j])-atan2(tempmaxy[k],tempmaxx[k])) - fabs(anglex[k]/2);
					inangley [j] = fabs(atan2(posy[j],posx[j])-atan2(tempmaxy1[k],tempmaxx1[k])) - fabs(angley[k]/2);
					if ((posz[j] > 3) || (centro[j]<5) || (inangley[j]>0))
					{
						posy2[j] = 1e10;
					}
					if ((posz[j] > 3) || (centro[j]<5) || (inanglex[j]>0))
					{
						posx2[j] = 1e10;
					}
				}
				sort(posy2,posy2+noa);
				sort(posx2,posx2+noa);
				double sumy = 0;
				double sumx = 0;
				for (j = 0; j < noave; j++){
					
					sumy += posy2[j];
					sumx += posx2[j];
				}
				a1[k] = sumx/noave;
				b1[k] = sumy/noave;
				cout<<endl<<endl<<"leading parial, "<<"k = "<<k<<", a = "<<a[k]<<",b = "<<b[k]<<endl;
				cout<<"trailing parial, "<<"k = "<<k<<", a = "<<a1[k]<<",b = "<<b1[k]<<endl;
				da[k] = a[k] - a1[k];
				db[k] = b[k] - b1[k];
				cout<<"seperation width, "<<"k = "<<k<<", da = "<<da[k]<<",db = "<<db[k]<<endl;
				 
				
			//clear files and put pointer back to the beginning to read in another image
			fin.clear( );
			fin.seekg( 0, std::ios::beg );
		}
		
		
		
		
		
		
		
			fin.close();
	
	}
    }



  // move atoms back inside simulation box and to new processors
  // use remap() instead of pbc() in case atoms moved a long distance
  // use irregular() in case atoms moved a long distance

  double **x = atom->x;
  tagint *image = atom->image;
  int nlocal = atom->nlocal;
  for (i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->reset_box();
  Irregular *irregular = new Irregular(lmp);
  irregular->migrate_atoms();
  delete irregular;
  if (domain->triclinic) domain->lamda2x(atom->nlocal);

  // check if any atoms were lost

  bigint natoms;
  bigint nblocal = atom->nlocal;
  MPI_Allreduce(&nblocal,&natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (natoms != atom->natoms  ) {
    char str[128];
    sprintf(str,"Lost atoms via displace_atoms: original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
    error->warning(FLERR,str);
  }
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of displace_atoms input line
------------------------------------------------------------------------- */

void DisplaceAtoms::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR,"Illegal displace_atoms command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal displace_atoms command");
      if (strcmp(arg[iarg+1],"box") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"lattice") == 0) scaleflag = 1;
      else error->all(FLERR,"Illegal displace_atoms command");
      iarg += 2;
    } else error->all(FLERR,"Illegal displace_atoms command");
  }
}
