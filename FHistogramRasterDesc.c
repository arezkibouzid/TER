/**************************************************************************************

						  F-HISTOGRAM COMPUTATION (RASTER DATA)

***************************************************************************************

									   Authors
									   -------
		L. Wendling (wendling@loria.fr), P. Matsakis (pmatsakis@cecs.missouri.edu).

									   Version
									   -------
								  2.01, March 2001.
					  For details (in French), see Section VII.

									   Based on
									   --------
	 [1] P. Matsakis, Relations spatiales structurelles et interpretation d'images,
		 Ph.D. dissertation, IRIT, Universite Paul Sabatier, Toulouse, France, 1998.
	 [2] P. Matsakis, L. Wendling, "A New Way to Represent the Relative Position of
		 Areal Objects", PAMI, vol. 21, no. 7, pp. 634-643, 1999.

---------------------------------------------------------------------------------------

This module provides the following functions: FRHistogram_CrispRaster(),
FRHistogram_FuzzyRaster(), F02Histogram_CrispRaster(), F02Histogram_FuzzyRaster(),
and readPGM(). The arguments accepted by these functions are described in Section I.
The functions themselves are described in Sections II. to VI. An example of main()
function is also presented at the very end of the file.

=======================================================================================
I. ARGUMENTS | These are the arguments accepted by the functions
			 | FRHistogram_CrispRaster(), FRHistogram_FuzzyRaster(),
			 | F02Histogram_CrispRaster(), F02Histogram_FuzzyRaster().
---------------------------------------------------------------------------------------

double typeForce
----------------
The type of force to be considered (i.e., any real number).
0.0 for constant forces, 2.0 for gravitational forces, etc.

int numberDirections
--------------------
It is a positive multiple of 4 (e.g., 16, 32, 64, 120, 180, 360).
Forces will be considered in 'numberDirections' directions.

double *histogram
-----------------
'numberDirections+1' values will be stored in the memory space pointed by 'histogram'.
For instance, 'histogram[0]' will be the resultant of forces in direction 0 ("to the
right"), and 'histogram[numberDirections]' the resultant of forces in direction
2PI (i.e., equal to 'histogram[0]'). 'histogram[numberDirections/4]' will be the
resultant of forces in direction PI/2 ("above") and 'histogram[numberDirections/2]'
the resultant of forces in direction PI ("to the left").

unsigned char *imageA
---------------------
The image that represents the argument object (sequential ordering, from top-left
to bottom-right). Gray levels are values between 0 and 255. The value 0 means that
the pixel does not belong at all to the object, and the value 255 that it totally
belongs to the object. The object is fuzzy if there exists a pixel whose gray
level g is such that 0<g<255. Otherwise, it is crisp.

unsigned char *imageB
---------------------
The image that represents the referent object.

int width, height
-----------------
The width and height of the two images.

int fuzzyMethod
---------------
For the computation of histograms associated with fuzzy objects only.
Either  1 (simple sum scheme, see Krishnapuram, Keller and Ma, 1993),
		2 (double sum scheme, see Dubois and Jaulent, 1987),
	   -1 (equivalent to the simple sum scheme, but pairs
		   of pixels are processed instead of alpha-cuts),
	   -2 (equivalent to the double sum scheme).

double p0, p1
-------------
For the computation of histograms of hybrid forces only. See [1].
'p0' is related to the size of the smallest object, and 'p1' to the size of
the intersection. If the objects are "not too close" (according to 'p0'), only
gravitational forces are exerted. When the objects intersect, or are "very close",
constant forces appear. We should have: 0<p0<p1. Typical values are p0=0.01 and p1=3.

=======================================================================================
II.

void FRHistogram_CrispRaster (double *histogram,
							  int numberDirections, double typeForce,
							  unsigned char *imageA, unsigned char *imageB,
							  int width, int height);

Computes the Fr-histogram associated
with a pair of crisp objects (raster data).
---------------------------------------------------------------------------------------
   WARNING | - The arguments are not checked.
		   | - The memory space pointed by 'histogram'
		   |   must be allocated by the calling function.
		   | - The objects are supposed to be crisp.
		   | - If 'typeForce==0.0', any pair (A,B) of objects
		   |   can be handled. However, if 'typeForce!=0.0",
		   |   only disjoint objects can currently be handled.
---------------------------------------------------------------------------------------
	  NOTE | In theory:
		   | - If A intersects B (one pixel is common to A and B),
		   |   (A,B) is assessable iff 'typeForce<1'.
		   | - If A is adjacent to B (no pixel is common to A and
		   |   B, but one pixel of A is an 8-neighbor of one pixel
		   |   of B), (A,B) is assessable iff 'typeForce<2'.
		   | - Otherwise (disjoint objects),
		   |   (A,B) is always assessable.

=======================================================================================
III.

void FRHistogram_FuzzyRaster (double *histogram,
							  int numberDirections, double typeForce,
							  unsigned char *imageA, unsigned char *imageB,
							  int width, int height,
							  int fuzzyMethod);

Computes the Fr-histogram associated
with a pair of fuzzy objects (raster data).
---------------------------------------------------------------------------------------
   WARNING | - The arguments are not checked.
		   | - The memory space pointed by 'histogram'
		   |   must be allocated by the calling function.
		   | - Currently, the only possible value for 'typeForce' is 0.0
		   |   (i.e., only histograms of constant forces can be computed),
		   |   and the only possible value for 'fuzzyMethod' is 1
		   |   (simple sum scheme).

=======================================================================================
IV.

void F02Histogram_CrispRaster (double *histogram, int numberDirections,
							   unsigned char *imageA, unsigned char *imageB,
							   int width, int height, double p0, double p1);

Computes the histogram of hybrid forces associated
with a pair of crisp objects (raster data).
---------------------------------------------------------------------------------------
   WARNING | - The arguments are not checked.
		   | - The memory space pointed by 'histogram'
		   |   must be allocated by the calling function.
		   | - The objects are supposed to be crisp.
---------------------------------------------------------------------------------------
	  NOTE | It is not possible to associate a histogram of gravitational
		   | forces with a pair of intersecting objects. The histogram of
		   | hybrid forces allows this limitation to be bypassed.

=======================================================================================
V.

void F02Histogram_FuzzyRaster (double *histogram, int numberDirections,
							   unsigned char *imageA, unsigned char *imageB,
							   int width, int height, double p0, double p1,
							   int fuzzyMethod);

Computes the histogram of hybrid forces associated
with a pair of fuzzy objects (raster data).
---------------------------------------------------------------------------------------
   WARNING | - The arguments are not checked.
		   | - The memory space pointed by 'histogram'
		   |   must be allocated by the calling function.
		   | - Currently, the only possible values for
		   |   'fuzzyMethod' are -1, 1 and 2.
---------------------------------------------------------------------------------------
	  NOTE | It is not possible to associate a histogram of gravitational
		   | forces with a pair of intersecting objects. The histogram of
		   | hybrid forces allows this limitation to be bypassed.

=======================================================================================
VI.

unsigned char* readPGM (char* filename, int* w, int* h)

Reads a PGM file.
---------------------------------------------------------------------------------------
  in  | The name of the PGM file.
  out | The width and height of the image.

  return | A pointer to an array of wxh unsigned char.
		 | NULL if the PGM file could not be read.

***************************************************************************/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/****************************/

#define  ZERO_FORCE_TYPE   0.001

#define  SIMPLE_CUTS      1
#define  DOUBLE_CUTS      2
#define  MIN_PIXELS      -1
#define  PRODUCT_PIXELS  -2

#define  DATA  unsigned char
#define  PI    3.141592653589

#define  absVal(x)     ((x)>0?(x):-(x))
#define  min255(x,y)   ((x)<(y)?(x)/255.0:(y)/255.0)
#define  sign(x)	   ((x)==0?0:((x)<0?-1:1))

// à optimiser via pointeurs quand le code tournera :-)
#define NBDesc          37 // +1 pour simplifier
#define NBMaxDir        256
/****************************/

struct Adresse
{
	DATA* adr;
};

/* Pour chainage des segments. */

struct Segment
{
	int x1;
	int y1;
	int x2;
	int y2;
	int Val;
	struct Segment* suivant;
};

/*******************************************************************************
							LOCAL FUNCTIONS
*******************************************************************************/

/*==============================================================
 Pointeurs sur lignes de l'image
 pour accelerer le traitement par la suite.
==============================================================*/

static void Cree_Tab_Pointeur(DATA* Image, struct Adresse* Tab,
	int Xsize, int Ysize)
{
	int i, j;
	i = Xsize * Ysize;
	for (j = 0; j < Ysize; j++)
	{
		i -= Xsize;
		Tab[j].adr = Image + i;
	}
}

/*==============================================================
 Creation d'un tableau ln de constantes (cas flou).
==============================================================*/

static void Cree_Tab_ln(double* Tab, int Xsize)
{
	int i;
	for (i = 1; i < Xsize; i++)
		*(Tab + i) = log((double)(i + 1) * (i + 1) / (i * (i + 2)));
}

/*==============================================================
 Bresenham suivant l'axe des X et sauvegarde
 dans structure chaine des changements.
==============================================================*/

static void Bresenham_X(int x1, int y1, int x2, int y2,
	int Borne_X, int* chaine)

{
	int Tmp, e, change, x, y, Delta_x, Delta_y, s1, s2, x_prec, y_prec;
	x = x1;
	y = y1;
	Delta_x = absVal(x2 - x1);
	Delta_y = absVal(y2 - y1);
	s1 = sign(x2 - x1);
	s2 = sign(y2 - y1);
	x_prec = x;
	y_prec = y;
	chaine[0] = 0;

	/* Permutation de delta_x et delta_y suivant la pente de seg. */
	if (Delta_y > Delta_x)
	{
		Tmp = Delta_x;
		Delta_x = Delta_y;
		Delta_y = Tmp;
		change = 1;
	}
	else
		change = 0;
	/* init. de e (cas inters. avec -1/2) */
	e = 2 * Delta_y - Delta_x;

	while (x < Borne_X)
	{
		while (e >= 0)
		{
			if (change)
			{
				x = x + s1;
				chaine[0]++;
				chaine[chaine[0]] = absVal(y - y_prec) + 1;
				chaine[0]++;
				chaine[chaine[0]] = 1;
				y_prec = y + 1;
			}
			else
			{
				y = y + s2;
				chaine[0]++;
				chaine[chaine[0]] = absVal(x - x_prec) + 1;
				chaine[0]++;
				chaine[chaine[0]] = 1;
				x_prec = x + 1;
			}
			e = e - 2 * Delta_x;
		}
		if (change)
			y = y + s2;
		else
			x = x + s1;
		e = e + 2 * Delta_y;
	}
	if (change)
		if (y_prec != y)
		{
			chaine[0]++;
			chaine[chaine[0]] = absVal(y - y_prec) + 1;
		}
		else {}
	else
		if (x_prec != x)
		{
			chaine[0]++;
			chaine[chaine[0]] = absVal(x - x_prec) + 1;
		}
}

/*==============================================================
 Meme chose mais suivant l'axe des Y.
==============================================================*/

static void Bresenham_Y(int x1, int y1, int x2, int y2,
	int Borne_Y, int* chaine)

{
	int Tmp, e, change, x, y, Delta_x, Delta_y, s1, s2, x_prec, y_prec;
	x = x1;
	y = y1;
	Delta_x = absVal(x2 - x1);
	Delta_y = absVal(y2 - y1);
	s1 = sign(x2 - x1);
	s2 = sign(y2 - y1);
	x_prec = x;
	y_prec = y;
	chaine[0] = 0;

	/* Permutation de delta_x et delta_y suivant la pente de seg. */
	if (Delta_y > Delta_x)
	{
		Tmp = Delta_x;
		Delta_x = Delta_y;
		Delta_y = Tmp;
		change = 1;
	}
	else
		change = 0;
	/* init. de e (cas inters. avec -1/2) */
	e = 2 * Delta_y - Delta_x;

	while (y < Borne_Y)
	{
		while (e >= 0)
		{
			if (change)
			{
				x = x + s1;
				chaine[0]++;
				chaine[chaine[0]] = absVal(y - y_prec) + 1;
				chaine[0]++;
				chaine[chaine[0]] = 1;
				y_prec = y + 1;
			}
			else
			{
				y = y + s2;
				chaine[0]++;
				chaine[chaine[0]] = absVal(x - x_prec) + 1;
				chaine[0]++;
				chaine[chaine[0]] = 1;
				x_prec = x + 1;
			}
			e = e - 2 * Delta_x;
		}
		if (change)
			y = y + s2;
		else
			x = x + s1;
		e = e + 2 * Delta_y;
	}
	if (change)
		if (y_prec != y)
		{
			chaine[0]++;
			chaine[chaine[0]] = absVal(y - y_prec) + 1;
		}
		else {}
	else
		if (x_prec != x)
		{
			chaine[0]++;
			chaine[chaine[0]] = absVal(x - x_prec) + 1;
		}
}

/*==============================================================
 On trace une ligne d'apres Bresenham.
 Decalage en X, et coupe de niveau.
==============================================================*/

static struct Segment* ligne_x(struct Adresse* Tab, int x, int y,
	int Borne_X, int Borne_Y,
	int* Chaine, int deb,
	int pas_x, int pas_y, int Cut)
{
	int i;
	int S;
	struct Segment* Liste_Seg, * Deb_liste, * Seg;
	int paspasser; /* pas bo ! a optimiser (gestion diff. des TWL)... */
	int xtmp, ytmp, xprec, yprec;
	paspasser = 1;
	S = 0;
	Liste_Seg = Deb_liste = NULL;

	while (((x != Borne_X) && (y != Borne_Y)) && (deb <= Chaine[0]))
	{
		i = 0;
		while ((i < Chaine[deb]) && ((x != Borne_X) && (y != Borne_Y)))
		{
			if (*(Tab[y].adr + x) >= Cut)
				if (!S)
				{
					S = 1;
					xtmp = xprec = x;
					ytmp = yprec = y;
				}
				else { xprec = x; yprec = y; }
			else
				if (S)
				{
					S = 0;
					Seg = (struct Segment*)malloc(sizeof(struct Segment));
					Seg->x1 = xtmp;
					Seg->y1 = ytmp;
					Seg->x2 = xprec;
					Seg->y2 = yprec;
					if (paspasser == 1)
					{
						Liste_Seg = Deb_liste = Seg;
						Liste_Seg->suivant = NULL;
						paspasser = 0;
					}
					else
					{
						Liste_Seg->suivant = Seg;
						Liste_Seg = Liste_Seg->suivant;
						Liste_Seg->suivant = NULL;
					}
				}
			i++;
			x += pas_x;
		}
		deb++;
		y += pas_y;
		deb++;
	}
	/* Segment limitrophe a la fenetre */
	if (S)
	{
		Seg = (struct Segment*)malloc(sizeof(struct Segment));
		Seg->x1 = xtmp;
		Seg->y1 = ytmp;
		Seg->x2 = xprec;
		Seg->y2 = yprec;
		if (paspasser == 1)
		{
			Liste_Seg = Deb_liste = Seg;
			Liste_Seg->suivant = NULL;
			paspasser = 0;
		}
		else
		{
			Liste_Seg->suivant = Seg;
			Liste_Seg = Liste_Seg->suivant;
			Liste_Seg->suivant = NULL;
		}
	}
	return(Deb_liste);
}

/*==============================================================
 On trace une ligne.
 Cas flou, decalage en X, sauvegarde pt a pt.
==============================================================*/

static struct Segment* ligne_x_floue(struct Adresse* Tab, int x, int y,
	int Borne_X, int Borne_Y,
	int* Chaine, int deb,
	int pas_x, int pas_y)
{
	int i;
	struct Segment* Liste_Seg, * Deb_liste, * Seg;
	int paspasser; /* pas bo ! a optimiser... */
	paspasser = 1;
	Liste_Seg = Deb_liste = NULL;

	while (((x != Borne_X) && (y != Borne_Y)) && (deb <= Chaine[0]))
	{
		i = 0;
		while ((i < Chaine[deb]) && ((x != Borne_X) && (y != Borne_Y)))
		{
			if (*(Tab[y].adr + x) != 0)
			{
				Seg = (struct Segment*)malloc(sizeof(struct Segment));
				Seg->x1 = x;
				Seg->y1 = y;
				Seg->x2 = x;
				Seg->y2 = y;
				Seg->Val = (int)*(Tab[y].adr + x);
				if (paspasser == 1)
				{
					Liste_Seg = Deb_liste = Seg;
					Liste_Seg->suivant = NULL;
					paspasser = 0;
				}
				else
				{
					Liste_Seg->suivant = Seg;
					Liste_Seg = Liste_Seg->suivant;
					Liste_Seg->suivant = NULL;
				}
			}
			i++;
			x += pas_x;
		}
		deb++;
		y += pas_y;
		deb++;
	}
	return(Deb_liste);
}

/*==============================================================
 Idem mais decalage en Y.
==============================================================*/

static struct Segment* ligne_y(struct Adresse* Tab, int x, int y,
	int Borne_X, int Borne_Y,
	int* Chaine, int deb,
	int pas_x, int pas_y, int Cut)
{
	int i;
	int S = 0;
	struct Segment* Liste_Seg, * Deb_liste, * Seg;
	int paspasser; /* pas bo ! a optimiser... */
	int xtmp, ytmp, xprec, yprec;
	paspasser = 1;
	Liste_Seg = Deb_liste = NULL;

	while (((x != Borne_X) && (y != Borne_Y)) && (deb <= Chaine[0]))
	{
		i = 0;
		while ((i < Chaine[deb]) && ((x != Borne_X) && (y != Borne_Y)))
		{
			if (*(Tab[y].adr + x) >= Cut)
				if (!S)
				{
					S = 1;
					xtmp = xprec = x;
					ytmp = yprec = y;
				}
				else { xprec = x; yprec = y; }
			else
				if (S)
				{
					S = 0;
					Seg = (struct Segment*)malloc(sizeof(struct Segment));
					/* permut. de x et y - projection suivant y - */
					Seg->x1 = ytmp;
					Seg->y1 = xtmp;
					Seg->x2 = yprec;
					Seg->y2 = xprec;
					if (paspasser == 1)
					{
						Liste_Seg = Deb_liste = Seg;
						Liste_Seg->suivant = NULL;
						paspasser = 0;
					}
					else
					{
						Liste_Seg->suivant = Seg;
						Liste_Seg = Liste_Seg->suivant;
						Liste_Seg->suivant = NULL;
					}
				}
			i++;
			y += pas_y;
		}
		deb++;
		x += pas_x;
		deb++;
	}
	/* Segment limitrophe a la fenetre */
	if (S)
	{
		Seg = (struct Segment*)malloc(sizeof(struct Segment));
		Seg->x1 = ytmp;
		Seg->y1 = xtmp;
		Seg->x2 = yprec;
		Seg->y2 = xprec;
		if (paspasser == 1)
		{
			Liste_Seg = Deb_liste = Seg;
			Liste_Seg->suivant = NULL;
			paspasser = 0;
		}
		else
		{
			Liste_Seg->suivant = Seg;
			Liste_Seg = Liste_Seg->suivant;
			Liste_Seg->suivant = NULL;
		}
	}
	return(Deb_liste);
}

/*==============================================================
 Cas flou, decalage en Y.
==============================================================*/

static struct Segment* ligne_y_floue(struct Adresse* Tab, int x, int y,
	int Borne_X, int Borne_Y,
	int* Chaine, int deb,
	int pas_x, int pas_y)
{
	int i;
	struct Segment* Liste_Seg, * Deb_liste, * Seg;
	int paspasser; /* pas bo ! a optimiser... */
	paspasser = 1;
	Liste_Seg = Deb_liste = NULL;

	while (((x != Borne_X) && (y != Borne_Y)) && (deb <= Chaine[0]))
	{
		i = 0;
		while ((i < Chaine[deb]) && ((x != Borne_X) && (y != Borne_Y)))
		{
			if (*(Tab[y].adr + x) != 0)
			{
				Seg = (struct Segment*)malloc(sizeof(struct Segment));
				/* permut. de x et y - projection suivant y - */
				Seg->x1 = y;
				Seg->y1 = x;
				Seg->x2 = y;
				Seg->y2 = x;
				Seg->Val = (int)*(Tab[y].adr + x);
				if (paspasser == 1)
				{
					Liste_Seg = Deb_liste = Seg;
					Liste_Seg->suivant = NULL;
					paspasser = 0;
				}
				else
				{
					Liste_Seg->suivant = Seg;
					Liste_Seg = Liste_Seg->suivant;
					Liste_Seg->suivant = NULL;
				}
			}
			i++;
			y += pas_y;
		}
		deb++;
		x += pas_x;
		deb++;
	}
	return(Deb_liste);
}

/*==============================================================
 Creation du Fr-histogramme, objets (nets) disjoints.
==============================================================*/

static void F0_disjoints(double* Histo,
	struct Segment* List_Seg_A,
	struct Segment* List_Seg_B,
	int Case1, int Case2)
{
	double Som_Seg_Pos;
	double Som_Seg_Neg;
	double d1, d2;

	struct Segment* Seg_A, * Seg_B, * pt_L_A, * pt_L_B;

	Som_Seg_Pos = Som_Seg_Neg = 0;

	pt_L_A = List_Seg_A;

	while (pt_L_A)
	{
		Seg_A = pt_L_A;
		d1 = absVal(Seg_A->x2 - Seg_A->x1) + 1;
		/* premier segment */
		pt_L_B = List_Seg_B;
		while (pt_L_B)
		{
			Seg_B = pt_L_B;
			d2 = absVal(Seg_B->x2 - Seg_B->x1) + 1;
			if (Seg_A->x2 < Seg_B->x1) Som_Seg_Pos += d1 * d2;
			else Som_Seg_Neg += d1 * d2;
			pt_L_B = pt_L_B->suivant;
		}
		pt_L_A = pt_L_A->suivant;
	}
	/* Attribution de la somme a l'histo. */
	Histo[Case1] += Som_Seg_Pos;
	/* Angle oppose. */
	Histo[Case2] += Som_Seg_Neg;
}

/*============================================================*/

static void F1_disjoints(double* Histo,
	struct Segment* List_Seg_A,
	struct Segment* List_Seg_B,
	int Case1, int Case2)
{
	double Som_Seg_Pos;
	double Som_Seg_Neg;
	double d1, d2, D;

	struct Segment* Seg_A, * Seg_B, * pt_L_A, * pt_L_B;

	Som_Seg_Pos = Som_Seg_Neg = 0;

	pt_L_A = List_Seg_A;

	while (pt_L_A)
	{
		Seg_A = pt_L_A;
		d1 = absVal(Seg_A->x2 - Seg_A->x1) + 1;
		/* premier segment */
		pt_L_B = List_Seg_B;
		while (pt_L_B)
		{
			Seg_B = pt_L_B;
			d2 = absVal(Seg_B->x2 - Seg_B->x1) + 1;
			/* d'abord => Pas de chevauchement */
			if (Seg_A->x2 < Seg_B->x1)
			{
				D = absVal(Seg_B->x1 - Seg_A->x2) - 1;
				Som_Seg_Pos += D * log(D) + (D + d1 + d2) * log(D + d1 + d2)
					- (D + d1) * log(D + d1) - (D + d2) * log(D + d2);
			}
			else
			{
				D = absVal(Seg_B->x2 - Seg_A->x1) - 1;
				Som_Seg_Neg += D * log(D) + (D + d1 + d2) * log(D + d1 + d2)
					- (D + d1) * log(D + d1) - (D + d2) * log(D + d2);
			}
			pt_L_B = pt_L_B->suivant;
		}
		pt_L_A = pt_L_A->suivant;
	}
	/* Attribution de la somme a l'histo. */
	Histo[Case1] += Som_Seg_Pos;
	/* Angle oppose. */
	Histo[Case2] += Som_Seg_Neg;
}

/*============================================================*/

static void F2_disjoints(double* Histo,
	struct Segment* List_Seg_A,
	struct Segment* List_Seg_B,
	int Case1, int Case2)
{
	double Som_Seg_Pos;
	double Som_Seg_Neg;
	double d1, d2, D;

	struct Segment* Seg_A, * Seg_B, * pt_L_A, * pt_L_B;

	Som_Seg_Pos = Som_Seg_Neg = 0;

	pt_L_A = List_Seg_A;

	while (pt_L_A)
	{
		Seg_A = pt_L_A;
		d1 = absVal(Seg_A->x2 - Seg_A->x1) + 1;
		/* premier segment */
		pt_L_B = List_Seg_B;
		while (pt_L_B)
		{
			Seg_B = pt_L_B;
			d2 = absVal(Seg_B->x2 - Seg_B->x1) + 1;
			if (Seg_A->x2 < Seg_B->x1)
			{
				D = absVal(Seg_B->x1 - Seg_A->x2) - 1;
				Som_Seg_Pos += log(((d1 + D) * (D + d2)) / (D * (d1 + D + d2)));
			}
			else
			{
				D = absVal(Seg_B->x2 - Seg_A->x1) - 1;
				Som_Seg_Neg += log(((d1 + D) * (D + d2)) / (D * (d1 + D + d2)));
			}
			pt_L_B = pt_L_B->suivant;
		}
		pt_L_A = pt_L_A->suivant;
	}
	/* Attribution de la somme a l'histo. */
	Histo[Case1] += Som_Seg_Pos;
	/* Angle oppose. */
	Histo[Case2] += Som_Seg_Neg;
}

/* Par hypothese : r!=0 et r!=1 et r!=2.
==============================================================*/

static void Fr_disjoints(double* Histo,
	struct Segment* List_Seg_A,
	struct Segment* List_Seg_B,
	int Case1, int Case2, double r)
{
	double Som_Seg_Pos;
	double Som_Seg_Neg;
	double d1, d2, D;

	struct Segment* Seg_A, * Seg_B, * pt_L_A, * pt_L_B;

	Som_Seg_Pos = Som_Seg_Neg = 0;

	pt_L_A = List_Seg_A;

	while (pt_L_A)
	{
		Seg_A = pt_L_A;
		d1 = absVal(Seg_A->x2 - Seg_A->x1) + 1;
		/* premier segment */
		pt_L_B = List_Seg_B;
		while (pt_L_B)
		{
			Seg_B = pt_L_B;
			d2 = absVal(Seg_B->x2 - Seg_B->x1) + 1;
			/* d'abord => Pas de chevauchement */
			if ((Seg_A->x2 == Seg_B->x2) && (Seg_A->x1 == Seg_B->x1))
			{
				D = absVal(Seg_A->x1 - Seg_A->x2) + 1;
				Som_Seg_Neg += (1.0 / ((1 - r) * (2 - r))) * pow(D, 2 - r);
				Som_Seg_Pos += (1.0 / ((1 - r) * (2 - r))) * pow(D, 2 - r);
			}
			else
				if (Seg_A->x2 < Seg_B->x1)
				{
					D = absVal(Seg_B->x1 - Seg_A->x2) - 1;
					Som_Seg_Pos += (1.0 / ((1 - r) * (2 - r))) *
						(pow(D, 2 - r) - pow(d1 + D, 2 - r) - pow(d2 + D, 2 - r) + pow(D + d1 + d2, 2 - r));
				}
				else
				{
					D = absVal(Seg_B->x2 - Seg_A->x1) - 1;
					Som_Seg_Neg += (1.0 / ((1 - r) * (2 - r))) *
						(pow(D, 2 - r) - pow(d1 + D, 2 - r) - pow(d2 + D, 2 - r) + pow(D + d1 + d2, 2 - r));
				}
			pt_L_B = pt_L_B->suivant;
		}
		pt_L_A = pt_L_A->suivant;
	}
	/* Attribution de la somme a l'histo. */
	Histo[Case1] += Som_Seg_Pos;
	/* Angle oppose. */
	Histo[Case2] += Som_Seg_Neg;
}

/*==============================================================
 Relation "before" pour forces hybrides F02.
==============================================================*/

static void Before(double* H, double x, double y, double z,
	double Y0, int Case, double Poids, double* Som_LN)
{
	if (Y0 <= y)
		(*Som_LN) += log((x + y) * (y + z) / (y * (x + y + z))) * Poids; /* Y0^2 en fac... */
	  /*H[Case].Val +=Y0*Y0*log((x+y)*(y+z)/(y*(x+y+z))) * Poids;  */
	else
		if (Y0 >= x + y + z)
			H[Case] += x * z * Poids;
		else
			if ((Y0 <= x + y) && (Y0 <= y + z))
				H[Case] += (Y0 * Y0 * log((y + z) * (x + y) / (Y0 * (x + y + z))) +
					(y - 3 * Y0) * (y - Y0) / 2) * Poids;
			else
				if ((Y0 >= x + y) && (Y0 <= y + z))
					H[Case] += (Y0 * Y0 * log((y + z) / (x + y + z)) - (x / 2 + y - 2 * Y0) * x) * Poids;
				else
					if (((Y0 >= x + y) && (Y0 >= y + z)) && (Y0 <= x + y + z))
						H[Case] += (Y0 * Y0 * log(Y0 / (x + y + z)) -
							(x + y + z - 3 * Y0) * (x + y + z - Y0) / 2 + x * z) * Poids;
					else
						/*   if ((Y0>=x+y)&&(Y0>=y+z)) */
						H[Case] += (Y0 * Y0 * log((y + x) / (x + y + z)) - (z / 2 + y - 2 * Y0) * z) * Poids;
}

/*==============================================================
 Relation "overlaps" pour forces hybrides F02.
==============================================================*/

static void Overlaps(double* H, double x, double y, double z,
	double Y0, int Case, double Poids)
{
	if ((Y0 <= x + y) && (Y0 <= y + z))
		H[Case] += (Y0 * Y0 * log((x + y) * (y + z) / (Y0 * (x + y + z))) -
			2 * Y0 * (y - 3 * Y0 / 4)) * Poids;
	else
		if (((Y0 <= x + y) && (y + z <= Y0)) && (0 <= y + z))
			H[Case] += (Y0 * Y0 * log((y + x) / (x + y + z)) + 2 * Y0 * z - (y + z) * (y + z) / 2) * Poids;
		else
			if (((x + y <= Y0) && (Y0 <= y + z)) && (0 <= x + y))
				H[Case] += (Y0 * Y0 * log((y + z) / (x + y + z)) + 2 * Y0 * x - (y + x) * (y + x) / 2) * Poids;
			else
				if (((x + y <= Y0) && (y + z <= Y0)) && (Y0 <= x + y + z))
					H[Case] += (Y0 * Y0 * log(Y0 / (x + y + z)) - (x + y + z - 3 * Y0) * (x + y + z - Y0) / 2 +
						x * z - y * y / 2) * Poids;
				else
					H[Case] += (x * z - y * y / 2) * Poids;
}

/*==============================================================
 Relation "overlapped by" pour forces hybrides F02.
==============================================================*/

static void Overlapped_By(double* H, double x, double y, double z,
	double Y0, int Case, double Poids)
{
	if (Y0 >= x + y + z)
		H[Case] += ((x + y + z) * (x + y + z) / 2) * Poids;
	else
		H[Case] += (Y0 * Y0 * log(Y0 / (x + y + z)) + 2 * Y0 * (x + y + z - 3 * Y0 / 4)) * Poids;
}

/*==============================================================
 Relation "contains" pour forces hybrides F02.
==============================================================*/

static void Contains(double* H, double x, double y, double z,
	double Y0, int Case, double Poids)
{
	if (Y0 <= x + y)
		H[Case] += (Y0 * Y0 * log((y + x) / (x + y + z)) + 2 * Y0 * z) * Poids;
	else
		if ((y + x <= Y0) && (Y0 <= x + y + z))
			H[Case] += (Y0 * Y0 * log(Y0 / (x + y + z)) + 2 * Y0 * z - (x + y - 3 * Y0)
				* (y + x - Y0) / 2) * Poids;
		else
			/*if (x+y+z<=Y0)*/
			H[Case] += (z / 2 + y + x) * z * Poids;
}

/*==============================================================
 Relation "during" pour forces hybrides F02.
==============================================================*/

static void During(double* H, double x, double y, double z,
	double Y0, int Case, double Poids)
{
	if (Y0 <= z + y)
		H[Case] += (Y0 * Y0 * log((y + z) / (x + y + z)) + 2 * Y0 * x) * Poids;
	else
		if ((y + z <= Y0) && (Y0 <= x + y + z))
			H[Case] += (Y0 * Y0 * log(Y0 / (x + y + z)) + 2 * Y0 * x - (z + y - 3 * Y0) *
				(y + z - Y0) / 2) * Poids;
		else
			/*if (x+y+z<=Y0)*/
			H[Case] += (x / 2 + y + z) * x * Poids;
}

/*==============================================================
 Forces hybrides F02, pour objets (nets) quelconques.
==============================================================*/

static void F02(double* Histo,
	struct Segment* List_Seg_A,
	struct Segment* List_Seg_B,
	int Case1, int Case2, double Poids, double Y0,
	double* Sum_LN_C1, double* Sum_LN_C2)
{
	double x, y, z;
	struct Segment* Seg_A, * Seg_B, * pt_L_A, * pt_L_B;

	pt_L_A = List_Seg_A;

	/* Attention a gerer avec les angles apres */
	while (pt_L_A)
	{
		/* premier segment */
		Seg_A = pt_L_A;

		pt_L_B = List_Seg_B;
		while (pt_L_B)
		{
			Seg_B = pt_L_B;
			/* d'abord Case 1 ... Optimisable */
			if (Seg_A->x1 <= Seg_B->x2)
			{
				if (Seg_A->x2 < Seg_B->x1)
				{
					x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
					z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
					y = absVal(Seg_B->x1 - Seg_A->x2) - 1;
					Before(Histo, x, y, z, Y0, Case1, Poids, Sum_LN_C1);
				}
				else
					if ((Seg_A->x2 < Seg_B->x2) && (Seg_A->x1 < Seg_B->x1))
					{
						x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
						z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
						y = -(absVal(Seg_A->x2 - Seg_B->x1) + 1);
						Overlaps(Histo, x, y, z, Y0, Case1, Poids);
					}
					else
						if ((Seg_B->x1 > Seg_A->x1) && (Seg_A->x2 >= Seg_B->x2))
						{
							x = absVal(Seg_B->x2 - Seg_A->x1) + 1;
							z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
							y = -z;
							Contains(Histo, x, y, z, Y0, Case1, Poids);
						}
						else
							if ((Seg_B->x1 <= Seg_A->x1) && (Seg_B->x2 > Seg_A->x2))
							{
								x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
								z = absVal(Seg_B->x2 - Seg_A->x1) + 1;
								y = -x;
								During(Histo, x, y, z, Y0, Case1, Poids);
							}
							else
							{
								x = z = absVal(Seg_B->x2 - Seg_A->x1) + 1;
								y = -x;
								Overlapped_By(Histo, x, y, z, Y0, Case1, Poids);
							}
			}

			/* droite dans l'autre sens attention au = */
			if (Seg_B->x1 <= Seg_A->x2)
			{
				if (Seg_B->x2 < Seg_A->x1)
				{
					x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
					z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
					y = absVal(Seg_A->x1 - Seg_B->x2) - 1;
					Before(Histo, x, y, z, Y0, Case2, Poids, Sum_LN_C2);
				}
				else
					if ((Seg_A->x2 > Seg_B->x2) && (Seg_A->x1 > Seg_B->x1))
					{
						x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
						z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
						y = -(absVal(Seg_B->x2 - Seg_A->x1) + 1);
						Overlaps(Histo, x, y, z, Y0, Case2, Poids);
					}
					else
						if ((Seg_A->x1 > Seg_B->x1) && (Seg_B->x2 >= Seg_A->x2))
						{
							x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
							z = absVal(Seg_A->x2 - Seg_B->x1) + 1;
							y = -x;
							During(Histo, x, y, z, Y0, Case2, Poids);
						}
						else
							if ((Seg_A->x1 <= Seg_B->x1) && (Seg_A->x2 > Seg_B->x2))
							{
								z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
								x = absVal(Seg_A->x2 - Seg_B->x1) + 1;
								y = -z;
								Contains(Histo, x, y, z, Y0, Case2, Poids);
							}
							else
							{
								x = z = absVal(Seg_A->x2 - Seg_B->x1) + 1;
								y = -z;
								Overlapped_By(Histo, x, y, z, Y0, Case2, Poids);
							}
			}

			pt_L_B = pt_L_B->suivant;
		}
		pt_L_A = pt_L_A->suivant;
	}
}

/*==============================================================
 Forces constantes F0, pour objets (nets) quelconques.
==============================================================*/

static void F0(double* Histo,
	struct Segment* List_Seg_A, struct Segment* List_Seg_B,
	int Case1, int Case2, double Poids)
{
	double x, y, z;
	struct Segment* Seg_A, * Seg_B, * pt_L_A, * pt_L_B;

	pt_L_A = List_Seg_A;

	/* Attention a gerer avec les angles apres */
	while (pt_L_A)
	{
		Seg_A = pt_L_A;
		/* premier segment */
		pt_L_B = List_Seg_B;
		while (pt_L_B)
		{
			Seg_B = pt_L_B;
			/* d'abord Case 1 ... Optimisable */
			if (Seg_A->x1 <= Seg_B->x2)
			{
				if (Seg_A->x2 < Seg_B->x1)
				{
					x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
					z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
					Histo[Case1] += Poids * x * z;
				}
				else
					if ((Seg_A->x2 < Seg_B->x2) && (Seg_A->x1 < Seg_B->x1))
					{
						x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
						z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
						y = -(absVal(Seg_A->x2 - Seg_B->x1) + 1);
						Histo[Case1] += Poids * (x * z - y * y / 2);
					}
					else
						if ((Seg_B->x1 > Seg_A->x1) && (Seg_A->x2 >= Seg_B->x2))
						{
							x = absVal(Seg_B->x2 - Seg_A->x1) + 1;
							z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
							y = -z;
							Histo[Case1] += Poids * (x + y + z / 2) * z;
						}
						else
							if ((Seg_B->x1 <= Seg_A->x1) && (Seg_B->x2 > Seg_A->x2))
							{
								x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
								z = absVal(Seg_B->x2 - Seg_A->x1) + 1;
								y = -x;
								Histo[Case1] += Poids * (x / 2 + y + z) * x;
							}
							else
							{
								x = z = absVal(Seg_B->x2 - Seg_A->x1) + 1;
								y = -x;
								Histo[Case1] += Poids * (x + y + z) * (x + y + z) / 2;
							}
			}

			/* droite dans l'autre sens attention au = */
			if (Seg_B->x1 <= Seg_A->x2)
			{
				if (Seg_B->x2 < Seg_A->x1)
				{
					x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
					z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
					Histo[Case2] += Poids * x * z;
				}
				else
					if ((Seg_A->x2 > Seg_B->x2) && (Seg_A->x1 > Seg_B->x1))
					{
						x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
						z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
						y = -(absVal(Seg_B->x2 - Seg_A->x1) + 1);
						Histo[Case2] += Poids * (x * z - y * y / 2);
					}
					else
						if ((Seg_A->x1 > Seg_B->x1) && (Seg_B->x2 >= Seg_A->x2))
						{
							x = absVal(Seg_A->x2 - Seg_A->x1) + 1;
							z = absVal(Seg_A->x2 - Seg_B->x1) + 1;
							y = -x;
							Histo[Case2] += Poids * (x / 2 + y + z) * x;
						}
						else
							if ((Seg_A->x1 <= Seg_B->x1) && (Seg_A->x2 > Seg_B->x2))
							{
								z = absVal(Seg_B->x2 - Seg_B->x1) + 1;
								x = absVal(Seg_A->x2 - Seg_B->x1) + 1;
								y = -z;
								Histo[Case2] += Poids * (x + y + z / 2) * z;
							}
							else
							{
								x = z = absVal(Seg_A->x2 - Seg_B->x1) + 1;
								y = -z;
								Histo[Case2] += Poids * (x + y + z) * (x + y + z) / 2;
							}
			}
			pt_L_B = pt_L_B->suivant;
		}
		pt_L_A = pt_L_A->suivant;
	}
}

/*==============================================================
 Forces hybrides F02, pour objets flous quelconques.
 Methode MIN_PIXELS (traitement des paires de pixels,
 min des membership grades).
==============================================================*/

static void F02_flous(double* Histo,
	struct Segment* List_Seg_A, struct Segment* List_Seg_B,
	int Case1, int Case2, double* Tab, double Y0,
	double* Sum_LN_C1, double* Sum_LN_C2)
{
	double x, z, y;

	struct Segment* Seg_A, * Seg_B, * pt_L_A, * pt_L_B;

	pt_L_A = List_Seg_A;

	/* Attention a gerer avec les angles apres */
	while (pt_L_A)
	{
		Seg_A = pt_L_A;
		/* premier segment */
		pt_L_B = List_Seg_B;
		while (pt_L_B)
		{
			Seg_B = pt_L_B;
			/* d'abord => Pas de chevauchement */
			if (Seg_A->x1 <= Seg_B->x1)
				if (Seg_A->x2 < Seg_B->x1)
				{
					y = absVal(Seg_B->x1 - Seg_A->x2) - 1;
					if (Y0 <= y)
						/*(*Sum_LN_C1) +=min255(Seg_A->Val,Seg_B->Val)*Tab[(int)y]; */
						Histo[Case1] += Y0 * Y0 * min255(Seg_A->Val, Seg_B->Val) * Tab[(int)y];
					else
					{
						x = 1; z = 1;
						Before(Histo, x, y, z, Y0, Case1,
							min255(Seg_A->Val, Seg_B->Val), Sum_LN_C1);
					}
					/* Temporaire
					//x=1;z=1;Histo[Case1]+=min255(Seg_A->Val,Seg_B->Val)*x*z; */
				}
				else
				{
					x = z = 1; y = -1;

					Overlapped_By(Histo, x, y, z, Y0, Case1, min255(Seg_A->Val, Seg_B->Val));
					Overlapped_By(Histo, x, y, z, Y0, Case2, min255(Seg_A->Val, Seg_B->Val));
					/* Histo[Case1]+=min255(Seg_A->Val,Seg_B->Val)*(x+y+z)*
					  (x+y+z)/2;
					Histo[Case2]+=min255(Seg_A->Val,Seg_B->Val)*(x+y+z)*
					  (x+y+z)/2; */
				}
			else
				if (Seg_B->x2 <= Seg_A->x1)
				{
					y = absVal(Seg_B->x1 - Seg_A->x2) - 1;
					if (Y0 <= y)
						(*Sum_LN_C2) += min255(Seg_A->Val, Seg_B->Val) * Tab[(int)y];
					/* Histo[Case2]+=Y0*Y0*min255(Seg_A->Val,Seg_B->Val)*Tab[(int)y]; */
					else
					{
						x = 1; z = 1;
						Before(Histo, x, y, z, Y0, Case2, min255(Seg_A->Val, Seg_B->Val), Sum_LN_C2);
					}
					/*Temporaire
					// x=1;z;Histo[Case2]+=min255(Seg_A->Val,Seg_B->Val)*x*z; */
				}
				else
				{
					/* cout << " on ne doit passer la... " << endl; */
					x = z = 1; y = -1;
					Overlapped_By(Histo, x, y, z, Y0, Case1, min255(Seg_A->Val, Seg_B->Val));
					Overlapped_By(Histo, x, y, z, Y0, Case2, min255(Seg_A->Val, Seg_B->Val));
					/* Histo[Case1]+=min255(Seg_A->Val,Seg_B->Val)*(x+y+z)*
					  (x+y+z)/2;
					Histo[Case2]+=min255(Seg_A->Val,Seg_B->Val)*(x+y+z)*(x+y+z)/2;*/
				}
			pt_L_B = pt_L_B->suivant;
		}
		pt_L_A = pt_L_A->suivant;
	}
}

/*==============================================================
 Choix de la methode a appliquer.
==============================================================*/

static void Choix_Methode(double* Histo,
	struct Segment* List_Seg_A,
	struct Segment* List_Seg_B,
	int Case1, int Case2, int methode, double Y0,
	double* Sum_LN_C1, double* Sum_LN_C2, double r)
{
	switch (methode)
	{
	case 2:
		if (fabs(r) <= ZERO_FORCE_TYPE)
			F0_disjoints(Histo, List_Seg_A, List_Seg_B, Case1, Case2);
		else if (fabs(r - 1) <= ZERO_FORCE_TYPE)
			F1_disjoints(Histo, List_Seg_A, List_Seg_B, Case1, Case2);
		else if (fabs(r - 2) <= ZERO_FORCE_TYPE)
			F2_disjoints(Histo, List_Seg_A, List_Seg_B, Case1, Case2);
		else
			Fr_disjoints(Histo, List_Seg_A, List_Seg_B, Case1, Case2, r);
		break;
	case 3:
		F02(Histo, List_Seg_A, List_Seg_B, Case1, Case2, 1.0, Y0, Sum_LN_C1, Sum_LN_C2);
		break;
	case 7:
		F0(Histo, List_Seg_A, List_Seg_B, Case1, Case2, 1.0);
		break;
	}
}

/*==============================================================
 Fr-histogrammes avec r quelconque mais objets disjoints.
 Prise en consideration du facteur multiplicatif
 qui depend de l'angle.
==============================================================*/

static void Angle_Histo(double* Histo, int Taille, double r)
{
	int case_dep, case_op, case_dep_neg, case_op_neg;
	double angle;
	double Pas_Angle = 2 * PI / Taille;
	case_dep = case_dep_neg = Taille / 2;
	case_op = 0;
	case_op_neg = Taille;
	angle = Pas_Angle;
	/* Case concernee */
	case_dep++;
	case_op++;
	case_dep_neg--;
	case_op_neg--;

	if (fabs(r) <= ZERO_FORCE_TYPE)
	{
		while (angle < PI / 4 + 0.0001)
		{
			Histo[case_dep] /= cos(angle);
			Histo[case_op] /= cos(angle);
			Histo[case_dep_neg] /= cos(angle);
			Histo[case_op_neg] /= cos(angle);
			angle += Pas_Angle;
			case_dep++;
			case_op++;
			case_dep_neg--;
			case_op_neg--;
		}
		while (angle < PI / 2)
		{
			Histo[case_dep] /= sin(angle);
			Histo[case_op] /= sin(angle);
			Histo[case_dep_neg] /= sin(angle);
			Histo[case_op_neg] /= sin(angle);
			angle += Pas_Angle;
			case_dep++;
			case_op++;
			case_dep_neg--;
			case_op_neg--;
		}
	}
	else if (fabs(r - 2) <= ZERO_FORCE_TYPE)
	{
		while (angle < PI / 4 + 0.0001)
		{
			Histo[case_dep] *= cos(angle);
			Histo[case_op] *= cos(angle);
			Histo[case_dep_neg] *= cos(angle);
			Histo[case_op_neg] *= cos(angle);
			angle += Pas_Angle;
			case_dep++;
			case_op++;
			case_dep_neg--;
			case_op_neg--;
		}
		while (angle < PI / 2)
		{
			Histo[case_dep] *= sin(angle);
			Histo[case_op] *= sin(angle);
			Histo[case_dep_neg] *= sin(angle);
			Histo[case_op_neg] *= sin(angle);
			angle += Pas_Angle;
			case_dep++;
			case_op++;
			case_dep_neg--;
			case_op_neg--;
		}
	}
	else if (fabs(r - 1) > ZERO_FORCE_TYPE)
	{
		while (angle < PI / 4 + 0.0001)
		{
			Histo[case_dep] /= pow(cos(angle), 1 - r);
			Histo[case_op] /= pow(cos(angle), 1 - r);
			Histo[case_dep_neg] /= pow(cos(angle), 1 - r);
			Histo[case_op_neg] /= pow(cos(angle), 1 - r);
			angle += Pas_Angle;
			case_dep++;
			case_op++;
			case_dep_neg--;
			case_op_neg--;
		}
		while (angle < PI / 2)
		{
			Histo[case_dep] /= pow(sin(angle), 1 - r);
			Histo[case_op] /= pow(sin(angle), 1 - r);
			Histo[case_dep_neg] /= pow(sin(angle), 1 - r);
			Histo[case_op_neg] /= pow(sin(angle), 1 - r);
			angle += Pas_Angle;
			case_dep++;
			case_op++;
			case_dep_neg--;
			case_op_neg--;
		}
	}
}

/*==============================================================
 Determination des segments suivant droite en X.
==============================================================*/

static void Calcul_Seg_X(struct Adresse* Tab_A, struct Adresse* Tab_B,
	double* Histo, int methode,
	int x1, int y1, int pas_x, int pas_y,
	int Xsize, int Ysize, int case_dep, int case_op,
	int* Chaine, int deb_chaine, int Cut[255],
	double* Tab_ln, double l,
	double* Sum_LN_C1, double* Sum_LN_C2, double r)
{
	int i, j, Prec_Cut, Prec_Cut_A, Prec_Cut_B;
	struct Segment* List_Seg_A, * List_Seg_B, * aux;

	switch (methode)
	{
		/* double sigma pixel a pixel */
	case 4: /* free a realiser ds le cas classique */
	  /* double sommation : cas pixel */
		List_Seg_A = ligne_x_floue(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y);
		List_Seg_B = ligne_x_floue(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y);
		if (List_Seg_A && List_Seg_B)
			F02_flous(Histo, List_Seg_A, List_Seg_B, case_dep, case_op,
				Tab_ln, l, Sum_LN_C1, Sum_LN_C2);
		while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
		while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
		break;

		/* Simple sommation cas flou */
	case 5:
		/* optimisable non trivial - gestion des segments */
		/* simple sommation (Style Khrisnapuram) - segment */
		i = 1; Prec_Cut = 0;
		while (i <= Cut[0])
		{
			List_Seg_A = ligne_x(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			List_Seg_B = ligne_x(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			if (List_Seg_A && List_Seg_B)
			{
				F02(Histo, List_Seg_A, List_Seg_B, case_dep, case_op,
					(double)(Cut[i] - Prec_Cut) / 255.0, l, Sum_LN_C1, Sum_LN_C2);
				Prec_Cut = Cut[i];
				i++;
			}
			else i = Cut[0] + 1;
			while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
			while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
		}
		break;

		/* double sommation segment a segment */
	case 6:
		/* optimisable mais non trivial */
		/* double sommation (Style Dubois) - segment */
		i = 1; Prec_Cut_A = 0;
		while (i <= Cut[0])
		{
			j = 1; Prec_Cut_B = 0;
			List_Seg_A = ligne_x(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			while (j <= Cut[0])
			{
				List_Seg_B = ligne_x(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[j]);
				if (List_Seg_A && List_Seg_B)
				{
					F02(Histo, List_Seg_A, List_Seg_B, case_dep, case_op,
						((double)(Cut[i] - Prec_Cut_A) / 255.0) *
					((double)(Cut[j] - Prec_Cut_B) / 255.0), l,
						Sum_LN_C1, Sum_LN_C2);
					Prec_Cut_B = Cut[j];
					j++;
				}
				else  j = Cut[0] + 1;
				while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
			}
			if (!List_Seg_A)
				i = Cut[0] + 1;
			else
			{
				Prec_Cut_A = Cut[i]; i++;
			}
			while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
		}
		break;

	case 8:
		/* optimisable non trivial - gestion des segments */
		/* simple sommation (Style Khrisnapuram) - segment */
		i = 1; Prec_Cut = 0;
		while (i <= Cut[0])
		{
			List_Seg_A = ligne_x(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			List_Seg_B = ligne_x(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			if (List_Seg_A && List_Seg_B)
			{
				F0(Histo, List_Seg_A, List_Seg_B, case_dep, case_op,
					(double)(Cut[i] - Prec_Cut) / 255.0);
				Prec_Cut = Cut[i];
				i++;
			}
			else i = Cut[0] + 1;
			while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
			while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
		}
		break;

		/* cas classiques... */
	default:
		List_Seg_A = ligne_x(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, 1);
		List_Seg_B = ligne_x(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, 1);
		if (List_Seg_A && List_Seg_B)
			Choix_Methode(Histo, List_Seg_A, List_Seg_B, case_dep, case_op,
				methode, l, Sum_LN_C1, Sum_LN_C2, r);
		while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
		while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
	}
}

/*==============================================================
 Determination des segments suivant droite en Y.
==============================================================*/

static void Calcul_Seg_Y(struct Adresse* Tab_A, struct Adresse* Tab_B,
	double* Histo, int methode,
	int x1, int y1, int pas_x, int pas_y,
	int Xsize, int Ysize, int case_dep, int case_op,
	int* Chaine, int deb_chaine,
	int Cut[255], double* Tab_ln, double l,
	double* Sum_LN_C1, double* Sum_LN_C2, double r)
{
	int i, j, Prec_Cut, Prec_Cut_A, Prec_Cut_B;
	struct Segment* List_Seg_A, * List_Seg_B, * aux;

	switch (methode)
	{
		/* double sigma pixel a pixel */
	case 4: /* free a realiser ds le cas classique */
	  /* double sommation : cas pixel */
		List_Seg_A = ligne_y_floue(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y);
		List_Seg_B = ligne_y_floue(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y);
		if (List_Seg_A && List_Seg_B)
			F02_flous(Histo, List_Seg_A, List_Seg_B, case_dep, case_op, Tab_ln, l,
				Sum_LN_C1, Sum_LN_C2);
		while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
		while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
		break;

		/* cas flou simple sigma */
	case 5:
		/* optimisable non trivial - gestion des segments */
		/* simple sommation (Style Khrisnapuram) - segment */
		i = 1; Prec_Cut = 0;
		while (i <= Cut[0])
		{
			List_Seg_A = ligne_y(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			List_Seg_B = ligne_y(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			if (List_Seg_A && List_Seg_B)
			{
				F02(Histo, List_Seg_A, List_Seg_B, case_dep, case_op,
					(double)(Cut[i] - Prec_Cut) / 255.0, l, Sum_LN_C1, Sum_LN_C2);
				Prec_Cut = Cut[i];
				i++;
			}
			else i = Cut[0] + 1;
			while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
			while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
		}
		break;

		/* cas flou double sigma */
	case 6:
		/* optimisable non trivial */
		/* double sommation (Style Dubois) - segment */
		i = 1; Prec_Cut_A = 0;
		while (i <= Cut[0])
		{
			List_Seg_A = ligne_y(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			j = 1; Prec_Cut_B = 0;
			while (j <= Cut[0])
			{
				List_Seg_B = ligne_y(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[j]);
				if (List_Seg_A && List_Seg_B)
				{
					F02(Histo, List_Seg_A, List_Seg_B, case_dep, case_op,
						((double)(Cut[i] - Prec_Cut_A) / 255.0) *
					((double)(Cut[j] - Prec_Cut_B) / 255.0), l,
						Sum_LN_C1, Sum_LN_C2);
					Prec_Cut_B = Cut[j];
					j++;
				}
				else j = Cut[0] + 1;
				while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
			}
			if (!List_Seg_A)
				i = Cut[0] + 1;
			else
			{
				Prec_Cut_A = Cut[i]; i++;
			}
			while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
		}
		break;

	case 8:
		/* optimisable non trivial - gestion des segments */
		/* simple sommation (Style Khrisnapuram) - segment */
		i = 1; Prec_Cut = 0;
		while (i <= Cut[0])
		{
			List_Seg_A = ligne_y(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			List_Seg_B = ligne_y(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, Cut[i]);
			if (List_Seg_A && List_Seg_B)
			{
				F0(Histo, List_Seg_A, List_Seg_B, case_dep, case_op,
					(double)(Cut[i] - Prec_Cut) / 255.0);
				Prec_Cut = Cut[i];
				i++;
			}
			else i = Cut[0] + 1;
			while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
			while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
		}
		break;

		/* cas classiques */
	default:
		List_Seg_A = ligne_y(Tab_A, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, 1);
		List_Seg_B = ligne_y(Tab_B, x1, y1, Xsize, Ysize, Chaine, deb_chaine, pas_x, pas_y, 1);
		if (List_Seg_A && List_Seg_B)
			Choix_Methode(Histo, List_Seg_A, List_Seg_B, case_dep, case_op,
				methode, l, Sum_LN_C1, Sum_LN_C2, r);
		while (List_Seg_A) { aux = List_Seg_A; List_Seg_A = List_Seg_A->suivant; free(aux); }
		while (List_Seg_B) { aux = List_Seg_B; List_Seg_B = List_Seg_B->suivant; free(aux); }
	}
}

/*======================================================================

	 Fonction principale
	 -------------------

	 methode 7 --> F0-histogramme  (objets nets quelconques)
	 methode 8 --> F0-histogramme  (objets flous quelconques,
									schema de la simple somme)
	 methode 2 --> Fr-histogramme  (objets nets disjoints)
	 methode 3 --> F02-histogramme (objets nets quelconques)
	 methode 4 --> F02-histogramme (objets flous quelconques,
									traitement par couples de pixels,
									produit des membership grades)
	 methode 5 --> F02-histogramme (objets flous quelconques,
									schema de la simple somme)
	 methode 6 --> F02-histogramme (objets flous quelconques,
									schema de la double somme)

=====================================================================*/

static void computeHistogram(double* Histo, int Taille, double typeForce,
	DATA* Image_A, DATA* Image_B,
	int Xsize, int Ysize,
	int methode, double p0, double p1)
{
	/* Structure pour l'image */
	struct Adresse* Tab_A, * Tab_B;
	double* tab_ln;
	int* Chaine;
	int deb_chaine, case_dep, case_dep_neg, case_op, case_op_neg;
	int x1, x2, y1, y2;
	double Sum_LN_C1, Sum_LN_C2;
	double angle;
	double Pas_Angle;
	int tempCut[256], Cut[256];
	DATA* ptrA, * ptrB;

	for (x1 = 0; x1 < 256; x1++) tempCut[x1] = Cut[x1] = 0;
	for (x1 = 0; x1 <= Taille; x1++) Histo[x1] = 0.0;
	x2 = y2 = y1 = 0;

	for (ptrA = Image_A, ptrB = Image_B, x1 = Xsize * Ysize - 1; x1 >= 0; x1--, ptrA++, ptrB++)
		if (*ptrA) {
			if (*ptrB) {
				y2 += (*ptrB); /* y2 is the area of B (times 255) */
				if (*ptrB < *ptrA) y1 += (*ptrB);
				else y1 += (*ptrA); /* y1 is the area of the intersection (times 255) */
			}
			x2 += (*ptrA); /* x2 is the area of A (times 255) */
			tempCut[(int)(*ptrA)] = 1;
		}
		else if (*ptrB) {
			y2 += (*ptrB);
			tempCut[(int)(*ptrB)] = 1;
		}

	for (x1 = 1; x1 < 256; x1++) if (tempCut[x1]) Cut[++Cut[0]] = x1;
	if (x2 > y2) p0 *= 2 * sqrt(y2 / (PI * 255));
	else p0 *= 2 * sqrt(x2 / (PI * 255));
	p1 *= 2 * sqrt(y1 / (PI * 255));
	if (p0 < p1) p0 = p1;

	Tab_A = (struct Adresse*)malloc(Ysize * sizeof(struct Adresse));
	Tab_B = (struct Adresse*)malloc(Ysize * sizeof(struct Adresse));

	Cree_Tab_Pointeur(Image_A, Tab_A, Xsize, Ysize);
	Cree_Tab_Pointeur(Image_B, Tab_B, Xsize, Ysize);

	/* Tableau de ln constante */
	tab_ln = (double*)malloc(Xsize * sizeof(double));

	Cree_Tab_ln(tab_ln, Xsize);

	Chaine = (int*)malloc((2 * Xsize + 1) * sizeof(int));

	Pas_Angle = 2 * PI / Taille; /* PI/(Taille+1) */

	/************* Angle = 0 *****************/
	angle = 0;
	x1 = y1 = y2 = 0;
	x2 = Xsize;
	Bresenham_X(x1, y1, x2, y2, Xsize, Chaine);
	case_dep = Taille / 2;
	case_op = 0;
	case_dep_neg = Taille / 2;
	case_op_neg = Taille;
	deb_chaine = 1;

	Sum_LN_C1 = Sum_LN_C2 = 0;

	for (y1 = 0; y1 < Ysize; y1++)
		Calcul_Seg_X(Tab_A, Tab_B, Histo, methode, x1, y1, 1, 1, Xsize, Ysize,
			case_dep, case_op, Chaine, deb_chaine, Cut, tab_ln, p0,
			&Sum_LN_C1, &Sum_LN_C2, typeForce);

	if (methode >= 3 && methode <= 6) /* F02 (flou ou pas) */
	{
		Histo[case_dep] = Histo[case_dep] / (p0 * p0) + Sum_LN_C1;
		Histo[case_op] = Histo[case_op] / (p0 * p0) + Sum_LN_C2;
	}

	/********** angle in [-pi/4,pi/4]-{0} ***************/
	/*      (projection suivant l'axe des X)            */

	angle += Pas_Angle;
	x2 = Xsize + 200;

	while (angle < PI / 4 + 0.0001) /* Arghhhhhh.... */
	{
		y2 = (int)(x2 * tan(angle));
		x1 = 0;
		y1 = 0;

		case_dep++;
		case_op++;

		/* On determine la droite a translater... */
		Bresenham_X(x1, y1, x2, y2, Xsize, Chaine);

		/* Vertical */
		deb_chaine = 1;

		Sum_LN_C1 = Sum_LN_C2 = 0;

		for (y1 = 0; y1 < Ysize; y1++)
			Calcul_Seg_X(Tab_A, Tab_B, Histo, methode, x1, y1, 1, 1, Xsize, Ysize,
				case_dep, case_op, Chaine, deb_chaine, Cut, tab_ln,
				p0 * cos(angle), &Sum_LN_C1, &Sum_LN_C2, typeForce);

		/* Horizontal */
		y1 = 0;
		while (x1 < Xsize)
		{
			x1 += Chaine[deb_chaine];
			deb_chaine += 2;
			Calcul_Seg_X(Tab_A, Tab_B, Histo, methode, x1, y1, 1, 1, Xsize, Ysize,
				case_dep, case_op, Chaine, deb_chaine, Cut, tab_ln,
				p0 * cos(angle), &Sum_LN_C1, &Sum_LN_C2, typeForce);
		}

		if (methode >= 3 && methode <= 6) /* F02 (flou ou pas) */
		{
			Histo[case_dep] = Histo[case_dep] / (cos(angle) * p0 * p0) +
				Sum_LN_C1 * cos(angle);
			Histo[case_op] = Histo[case_op] / (cos(angle) * p0 * p0) +
				Sum_LN_C2 * cos(angle);
		}

		/************* Angle negatif oppose *******/
		case_dep_neg--;
		case_op_neg--;

		/* Vertical */
		deb_chaine = 1;

		Sum_LN_C1 = Sum_LN_C2 = 0;

		x1 = 0;

		/*for (y1=0;y1<Ysize;y1++) // AR Ysize...
	   Calcul_Seg_X(Tab_A, Tab_B, Histo, methode, x1, y1, 1, -1, Xsize, -1,
			  case_op_neg, case_dep_neg, Chaine, deb_chaine, Cut,
			  tab_ln, p0*cos(angle), &Sum_LN_C1,&Sum_LN_C2,typeForce);

		 // Horizontal
		 deb_chaine=1;
		 y1=Ysize-1;
		 x1=Xsize;
		  while (x1>0)
	  {
		 x1-=Chaine[deb_chaine];
		 deb_chaine+=2;
		 Calcul_Seg_X(Tab_A, Tab_B, Histo, methode, x1, y1, 1, -1, -1, Ysize,
				case_op_neg, case_dep_neg, Chaine, deb_chaine, Cut,
				tab_ln, p0*cos(angle),&Sum_LN_C1,&Sum_LN_C2,typeForce);
	  }
	  */
		if (case_dep_neg >= (Taille / 2 - Taille / 8))
		{
			x1 = 0;
			for (y1 = 0; y1 < Ysize; y1++) /* AR Ysize... */
				Calcul_Seg_X(Tab_A, Tab_B, Histo, methode, x1, y1, 1, -1, Xsize, -1,
					case_dep_neg, case_op_neg, Chaine, deb_chaine, Cut,
					tab_ln, p0 * cos(angle), &Sum_LN_C1, &Sum_LN_C2, typeForce);

			/* Horizontal */
			deb_chaine = 1;
			y1 = Ysize - 1;
			while (x1 < Xsize)
			{
				x1 += Chaine[deb_chaine];
				deb_chaine += 2;
				Calcul_Seg_X(Tab_A, Tab_B, Histo, methode, x1, y1, 1, -1, Xsize, -1,
					case_dep_neg, case_op_neg, Chaine, deb_chaine, Cut,
					tab_ln, p0 * cos(angle), &Sum_LN_C1, &Sum_LN_C2, typeForce);
			}

			if (methode >= 3 && methode <= 6) /* F02 (flou ou pas) */
			{
				Histo[case_dep_neg] = Histo[case_dep_neg] / (cos(angle) * p0 * p0) +
					Sum_LN_C1 * cos(angle);
				Histo[case_op_neg] = Histo[case_op_neg] / (cos(angle) * p0 * p0) +
					Sum_LN_C2 * cos(angle);
			}

			angle += Pas_Angle;
		}
		else
		{
			x1 = 0;
			deb_chaine = 1;

			for (y1 = 0; y1 < Ysize; y1++) /* AR Ysize...*/
				Calcul_Seg_X(Tab_A, Tab_B, Histo, methode, x1, y1, 1, -1, Xsize, -1,
					case_dep_neg, case_op_neg, Chaine, deb_chaine, Cut,
					tab_ln, p0 * cos(angle), &Sum_LN_C1,
					&Sum_LN_C2, typeForce);

			/* Horizontal */
			deb_chaine = 1;
			y1 = Ysize - 1;
			while (x1 < Xsize)
			{
				x1 += Chaine[deb_chaine];
				deb_chaine += 2;
				Calcul_Seg_X(Tab_A, Tab_B, Histo, methode, x1, y1, 1, -1, Xsize,
					-1, case_dep_neg, case_op_neg, Chaine, deb_chaine,
					Cut, tab_ln, p0 * cos(angle), &Sum_LN_C1, &Sum_LN_C2, typeForce);
			}

			if (methode >= 3 && methode <= 6) /* F02 (flou ou pas) */
			{
				Histo[case_dep_neg] = Histo[case_dep_neg] / (cos(angle) * p0 * p0) +
					Sum_LN_C1 * cos(angle);
				Histo[case_op_neg] = Histo[case_op_neg] / (cos(angle) * p0 * p0) +
					Sum_LN_C2 * cos(angle);
			}
			angle += Pas_Angle;
		}
	}

	/*********** angle in ]-PI/2,-PI/4[ or ]PI/4,PI/2[ ***************/
	/*              (projection suivant l'axe des Y)                 */

	while (angle < PI / 2 - 0.0001)   /* another Aaaarrggggghhh....... */
	{
		y2 = (int)(x2 * tan(angle));
		x1 = 0;
		y1 = 0;
		case_dep++;
		case_op++;

		/* On determine la droite a translater... */
		Bresenham_Y(x1, y1, x2, y2, Ysize, Chaine);

		/* Horizontal */
		Sum_LN_C1 = Sum_LN_C2 = 0;

		deb_chaine = 1;
		for (x1 = 0; x1 < Xsize; x1++)
			Calcul_Seg_Y(Tab_A, Tab_B, Histo, methode, x1, y1, 1, 1, Xsize,
				Ysize, case_dep, case_op, Chaine, deb_chaine, Cut, tab_ln,
				p0 * sin(angle), &Sum_LN_C1, &Sum_LN_C2, typeForce);

		/* Vertical */
		x1 = 0;
		y1 = 0;
		while (y1 < Ysize)
		{
			y1 += Chaine[deb_chaine];
			deb_chaine += 2;
			Calcul_Seg_Y(Tab_A, Tab_B, Histo, methode, x1, y1, 1, 1, Xsize, Ysize,
				case_dep, case_op, Chaine, deb_chaine, Cut, tab_ln,
				p0 * sin(angle), &Sum_LN_C1, &Sum_LN_C2, typeForce);
		}

		if (methode >= 3 && methode <= 6) /* F02 (flou ou pas) */
		{
			Histo[case_dep] = Histo[case_dep] / (sin(angle) * p0 * p0) +
				Sum_LN_C1 * sin(angle);
			Histo[case_op] = Histo[case_op] / (sin(angle) * p0 * p0) +
				Sum_LN_C2 * sin(angle);
		}

		/******** Partie oppose ***************/
		case_dep_neg--;
		case_op_neg--;

		/* Horizontal */
		Sum_LN_C1 = Sum_LN_C2 = 0;
		deb_chaine = 1;
		y1 = 0;  /*  y1=Ysize-1 est envisageable aussi... */

		for (x1 = 0; x1 < Xsize; x1++)
			Calcul_Seg_Y(Tab_A, Tab_B, Histo, methode, x1, y1, -1, 1, -1, Ysize,
				case_op_neg, case_dep_neg, Chaine, deb_chaine, Cut, tab_ln,
				p0 * sin(angle), &Sum_LN_C2, &Sum_LN_C1, typeForce);

		/* Vertical */
		x1 = Xsize - 1;
		y1 = 0;
		while (y1 < Ysize)
		{
			y1 += Chaine[deb_chaine];
			deb_chaine += 2;
			Calcul_Seg_Y(Tab_A, Tab_B, Histo, methode, x1, y1, -1, 1, -1, Ysize,
				case_op_neg, case_dep_neg, Chaine, deb_chaine, Cut, tab_ln,
				p0 * sin(angle), &Sum_LN_C2, &Sum_LN_C1, typeForce);
		}

		if (methode >= 3 && methode <= 6) /* F02 (flou ou pas) */
		{
			Histo[case_dep_neg] = Histo[case_dep_neg] / (sin(angle) * p0 * p0) +
				Sum_LN_C1 * sin(angle);
			Histo[case_op_neg] = Histo[case_op_neg] / (sin(angle) * p0 * p0) +
				Sum_LN_C2 * sin(angle);
		}

		angle += Pas_Angle;
	}

	/************* Angle = PI/2 *****************/
	y1 = x1 = x2 = 0;
	y2 = Ysize;
	case_dep++;
	case_op++;
	Bresenham_Y(x1, y1, x2, y2, Ysize, Chaine);

	deb_chaine = 1;
	Sum_LN_C1 = Sum_LN_C2 = 0;

	for (x1 = 0; x1 < Xsize; x1++)
		Calcul_Seg_Y(Tab_A, Tab_B, Histo, methode, x1, y1, 1, 1, Xsize,
			Ysize, case_dep, case_op, Chaine, deb_chaine,
			Cut, tab_ln, p0, &Sum_LN_C1, &Sum_LN_C2, typeForce);

	if (methode >= 3 && methode <= 6) /* F02 (flou ou pas) */
	{
		Histo[case_dep] = Histo[case_dep] / (p0 * p0) + Sum_LN_C1;
		Histo[case_op] = Histo[case_op] / (p0 * p0) + Sum_LN_C2;
	}

	/* Atribution de la valeur associee a -PI */
	Histo[Taille] += Histo[0];
	Histo[0] = Histo[Taille];

	if (methode < 3 || methode>6) /* Fr quelconque (mais pas hybride) */
		Angle_Histo(Histo, Taille, typeForce);

	free(Tab_A);
	free(Tab_B);
	free(tab_ln);
	free(Chaine);
}

/*******************************************************************************
							GLOBAL FUNCTIONS
*******************************************************************************/

/*=====================================================================================
  extractSubImage | Extracts a sub-image from an image.
---------------------------------------------------------------------------------------
  in  | - The image, its width and height.
	  | - The minimum and maximum coordinates of the
	  |   image pixels that also belong to the sub-image.
  out | The sub-image.

  'extractSubImage' assumes that '(xmax-xmin+1)*(ymax-ymin+1)' bytes have been
   allocated for the sub-image. It does not check the arguments. In particular,
   it assumes that 'xmin' and 'ymin' are greater than or equal to 0; 'xmax' is
   lower than 'width', and 'ymax' is lower than 'height'.
---------------------------------------------------------------------------------------
   WARNING | This method is not used by the module and has not been tested.
=====================================================================================*/

void extractSubImage(unsigned char* image, int width, int height,
	int xmin, int ymin, int xmax, int ymax,
	unsigned char* subimage) {
	int x, y;
	unsigned char* ptr, * ptrI;

	ptr = subimage;
	for (ptrI = image, y = 0; y < ymin; y++, ptrI += width);
	for (; y <= ymax; y++) {
		ptrI += xmin;
		for (x = xmin; x <= xmax; x++, ptr++, ptrI++) *ptr = *ptrI;
		ptrI += width - ymax - 1;
	}
}

/*=====================================================================================
  rotateImage | Applies a 90 degrees angle rotation to the image (counterclockwise).
---------------------------------------------------------------------------------------
  in  | The image, its width and height.
  out | The rotated image.

  'rotateImage' assumes that 'width*height' bytes have been
   allocated for the rotated image. It does not check the arguments.
=====================================================================================*/

void rotateImage(unsigned char* image, int width, int height,
	unsigned char* rotatedImage) {
	int x, y;
	unsigned char* ptr, * ptrI;

	ptr = rotatedImage;
	for (x = width - 1, ptrI = image + x; x >= 0; x--, ptrI = image + x)
		for (y = 0; y < height; y++, ptrI += width, ptr++) *ptr = *ptrI;
}

/*=====================================================================================
   readPGM | Reads a PGM file.
---------------------------------------------------------------------------------------
   This function is described at the beginning of the file.
=====================================================================================*/

unsigned char* readPGM(char* filename, int* w, int* h) {
	FILE* file;
	char line[256];
	unsigned char* data;
	int i, int_tmp, aux, binary;

	/* Opening file.
	----------------*/
	if ((file = fopen(filename, "r")) == NULL) {
		/* ERROR: could not open the file */
		*h = *w = 0;
		return(NULL);
	}

	/* Is the PGM file a binary file or a text file?
	------------------------------------------------*/
	fgets(line, 256, file);
	if (strncmp(line, "P5", 2)) {
		if (strncmp(line, "P2", 2)) {
			/* ERROR: not a pgm file */
			*h = *w = 0;
			return(NULL);
		}
		else binary = 0;
	}
	else binary = 1;

	/* Skipping comment lines,
	   reading width, height and maximum value.
	-------------------------------------------*/
	fgets(line, 256, file);
	while (line[0] == '#') fgets(line, 256, file);
	sscanf(line, "%d %d", w, h);
	fgets(line, 256, file);
	sscanf(line, "%d", &aux);
	aux = (*w) * (*h);

	/* Loading data.
	----------------*/
	data = (unsigned char*)malloc(aux * sizeof(unsigned char));
	if (binary)
		fread((void*)data, sizeof(unsigned char), aux, file);
	else
		for (i = 0; i < aux; i++) {
			fscanf(file, "%d", &int_tmp);
			data[i] = int_tmp;
		}

	fclose(file);
	return(data);
}

/*==============================================================================
   FRHistogram_CrispRaster | Computation of the Fr-histogram
							 associated with a pair of crisp objects
							 (raster data).
--------------------------------------------------------------------------------
   This function is described at the beginning of the file.
==============================================================================*/

void FRHistogram_CrispRaster(double* histogram,
	int numberDirections, double typeForce,
	unsigned char* imageA, unsigned char* imageB,
	int width, int height) {
	int methode;
	double p0, p1;

	if (fabs(typeForce) <= ZERO_FORCE_TYPE) methode = 7;
	else methode = 2;
	p0 = 0.01; /* doesn't matter here */
	p1 = 3.0;  /* doesn't matter here */

	if (width >= height)
		computeHistogram(histogram,
			numberDirections, typeForce,
			imageA, imageB, width, height,
			methode, p0, p1);

	else { /* Currently, 'computeHistogram' assumes that
			  'width' is greater than or equal to 'height. */

		int i, j;
		unsigned char* rotImageA;
		unsigned char* rotImageB;
		double* auxHistogram;

		rotImageA = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		rotImageB = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		auxHistogram = (double*)malloc((numberDirections + 1) * sizeof(double));

		rotateImage(imageA, width, height, rotImageA);
		rotateImage(imageB, width, height, rotImageB);
		computeHistogram(auxHistogram,
			numberDirections, typeForce,
			rotImageA, rotImageB, height, width,
			methode, p0, p1);

		for (i = 0, j = numberDirections / 4; i <= numberDirections; i++, j++)
			histogram[i] = auxHistogram[j % numberDirections];

		free(rotImageA);
		free(rotImageB);
		free(auxHistogram);
	}
}

/*==============================================================================
   FRHistogram_FuzzyRaster | Computation of the Fr-histogram
							 associated with a pair of fuzzy objects
							 (raster data).
--------------------------------------------------------------------------------
   This function is described at the beginning of the file.
==============================================================================*/

void FRHistogram_FuzzyRaster(double* histogram,
	int numberDirections, double typeForce,
	unsigned char* imageA, unsigned char* imageB,
	int width, int height,
	int fuzzyMethod) {
	double p0, p1;
	int methode;

	p0 = 0.01; /* doesn't matter here */
	p1 = 3.0;  /* doesn't matter here */

	switch (fuzzyMethod) {
	case SIMPLE_CUTS: /* only possibility right now */
	default: methode = 8;
	}

	if (width >= height)
		computeHistogram(histogram,
			numberDirections, typeForce,
			imageA, imageB, width, height,
			methode, p0, p1);

	else { /* Currently, 'computeHistogram' assumes that
			  'width' is greater than or equal to 'height. */

		int i, j;
		unsigned char* rotImageA;
		unsigned char* rotImageB;
		double* auxHistogram;

		rotImageA = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		rotImageB = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		auxHistogram = (double*)malloc((numberDirections + 1) * sizeof(double));

		rotateImage(imageA, width, height, rotImageA);
		rotateImage(imageB, width, height, rotImageB);
		computeHistogram(auxHistogram,
			numberDirections, typeForce,
			rotImageA, rotImageB, height, width,
			methode, p0, p1);

		for (i = 0, j = numberDirections / 4; i <= numberDirections; i++, j++)
			histogram[i] = auxHistogram[j % numberDirections];

		free(rotImageA);
		free(rotImageB);
		free(auxHistogram);
	}
}

/*==============================================================================
   F02Histogram_CrispRaster | Computation of the histogram of hybrid forces
							  associated with a pair of crisp objects
							  (raster data).
--------------------------------------------------------------------------------
   This function is described at the beginning of the file.
==============================================================================*/

void F02Histogram_CrispRaster(double* histogram, int numberDirections,
	unsigned char* imageA, unsigned char* imageB,
	int width, int height, double p0, double p1) {
	int methode;
	double typeForce;

	methode = 3;
	typeForce = 2.0; /* doesn't matter here */

	if (width >= height)
		computeHistogram(histogram,
			numberDirections, typeForce,
			imageA, imageB, width, height,
			methode, p0, p1);

	else { /* Currently, 'computeHistogram' assumes that
			  'width' is greater than or equal to 'height. */

		int i, j;
		unsigned char* rotImageA;
		unsigned char* rotImageB;
		double* auxHistogram;

		rotImageA = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		rotImageB = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		auxHistogram = (double*)malloc((numberDirections + 1) * sizeof(double));

		rotateImage(imageA, width, height, rotImageA);
		rotateImage(imageB, width, height, rotImageB);
		computeHistogram(auxHistogram,
			numberDirections, typeForce,
			rotImageA, rotImageB, height, width,
			methode, p0, p1);

		for (i = 0, j = numberDirections / 4; i <= numberDirections; i++, j++)
			histogram[i] = auxHistogram[j % numberDirections];

		free(rotImageA);
		free(rotImageB);
		free(auxHistogram);
	}
}

/*==============================================================================
   F02Histogram_FuzzyRaster | Computation of the histogram of hybrid forces
							  associated with a pair of fuzzy objects
							  (raster data).
--------------------------------------------------------------------------------
   This function is described at the beginning of the file.
==============================================================================*/

void F02Histogram_FuzzyRaster(double* histogram, int numberDirections,
	unsigned char* imageA, unsigned char* imageB,
	int width, int height, double p0, double p1,
	int fuzzyMethod) {
	int methode;
	double typeForce;

	typeForce = 2.0; /* doesn't matter here */
	switch (fuzzyMethod) {
	case MIN_PIXELS: methode = 4; break;
	case DOUBLE_CUTS: methode = 6; break;
	case SIMPLE_CUTS: /* default value */
	default: methode = 5; break;
	}

	if (width >= height)
		computeHistogram(histogram,
			numberDirections, typeForce,
			imageA, imageB, width, height,
			methode, p0, p1);

	else { /* Currently, 'computeHistogram' assumes that
			  'width' is greater than or equal to 'height. */

		int i, j;
		unsigned char* rotImageA;
		unsigned char* rotImageB;
		double* auxHistogram;

		rotImageA = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		rotImageB = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		auxHistogram = (double*)malloc((numberDirections + 1) * sizeof(double));

		rotateImage(imageA, width, height, rotImageA);
		rotateImage(imageB, width, height, rotImageB);
		computeHistogram(auxHistogram,
			numberDirections, typeForce,
			rotImageA, rotImageB, height, width,
			methode, p0, p1);

		for (i = 0, j = numberDirections / 4; i <= numberDirections; i++, j++)
			histogram[i] = auxHistogram[j % numberDirections];

		free(rotImageA);
		free(rotImageB);
		free(auxHistogram);
	}
}

/*==============================================================================
 F02Histogram_CrispRaster | Computation of the histogram of hybrid forces
 associated with a pair of crisp objects
 (raster data).
 --------------------------------------------------------------------------------
 This function is described at the beginning of the file.
 ==============================================================================*/

void PhiDesc_Histogram(double PhiDesc[NBDesc][NBMaxDir], int numberDirections,
	unsigned char* imageA, unsigned char* imageB,
	int width, int height) {
	int methode;
	double typeForce;
	double* auxHistogram;
	auxHistogram = (double*)malloc((numberDirections + 1) * sizeof(double));
	double p0, p1;
	methode = 3;
	typeForce = 2.0; /* doesn't matter here */
	p0 = 0.01; /* doesn't matter here */
	p1 = 3.0;  /* doesn't matter here */

	if (width >= height)
		computeHistogram(auxHistogram,
			numberDirections, typeForce,
			imageA, imageB, width, height,
			methode, p0, p1);

	else { /* Currently, 'computeHistogram' assumes that
			'width' is greater than or equal to 'height. */

		int i, j;
		unsigned char* rotImageA;
		unsigned char* rotImageB;
		//double *auxHistogram;

		rotImageA = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		rotImageB = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		//auxHistogram=(double *)malloc((numberDirections+1)*sizeof(double));

		rotateImage(imageA, width, height, rotImageA);
		rotateImage(imageB, width, height, rotImageB);
		computeHistogram(auxHistogram,
			numberDirections, typeForce,
			rotImageA, rotImageB, height, width,
			methode, p0, p1);

		//for(i=0,j=numberDirections/4;i<=numberDirections;i++,j++)
		//    histogram[i]=auxHistogram[j%numberDirections];

		free(rotImageA);
		free(rotImageB);
	}
	free(auxHistogram);
}

/*******************************************************************************
							EXAMPLE OF main() FUNCTION
*******************************************************************************/

/*==============================================================================
   main | This function is provided as an example and for testing purposes.
--------------------------------------------------------------------------------
   Compile (e.g., "gcc FHistogramRaster.c -lm"), execute ("a.out"),
   and follow the instructions.
==============================================================================*/

int main() {
	FILE* histogramFile;
	unsigned char* imageA, * imageB;
	char nameImageA[100], nameImageB[100], nameHistogramFile[100];
	int i, j, widthA, heightA, widthB, heightB, numberOfDirections, fuzzyMethod;
	double* histogram, typeForce, p0, p1;
	double PhiDesc[NBDesc][NBMaxDir];
	int choice;
	double rdeb;
	double rfin;
	double rpas; // début interval et fin peut être exclus
	//float vardeb;

	/* Choosing the function and the arguments.
	-------------------------------------------*/

	/*printf("\n1. Fr-Histogram, crisp objects.");
	printf("\n2. Fr-Histogram, fuzzy objects.");
	printf("\n3. Histogram of hybrid forces, crisp objects.");
	printf("\n4. Histogram of hybrid forces, fuzzy objects.");
	printf("\n5. Phi descriptor, crisp objects.");
	printf("\n\nYour choice ? ");
	scanf("%d",&choice);
	if(choice<1 || choice>5) exit(0);
	 */

	choice = 5; // on fixe à r qui varie

	printf("\nEnter arguments.");
	printf("\nYou are supposed to know the types and domains.");

	printf("\n\nHistogram will be stored in file: ");
	scanf("%s", nameHistogramFile);
	if ((histogramFile = fopen(nameHistogramFile, "wt")) == NULL)
	{
		printf("\nERROR histogram file\n\n"); exit(1);
	}

	printf("Number of directions to be considered is: ");
	scanf("%d", &numberOfDirections);

	// if(choice==1 || choice==2) {
	printf("Type of force: start: ");
	scanf("%lf", &rdeb);
	printf("Type of force: end: ");
	scanf("%lf", &rfin);
	printf("Type of force: step: ");
	scanf("%lf", &rpas);

	// intitialisation à deb
	//typeforce=rdeb;

	//printf("Type of force is: ");
	//scanf("%lf",&typeForce);

	//}
	printf("Argument object is defined by the PGM image: ");
	scanf("%s", nameImageA);
	printf("Referent object is defined by the PGM image: ");
	scanf("%s", nameImageB);
	if (choice == 2 || choice == 4) {
		printf("Fuzzy method is: ");
		scanf("%d", &fuzzyMethod);
	}
	if (choice == 3 || choice == 4) {
		printf("Parameter p0 is: ");
		scanf("%lf", &p0);
		printf("Parameter p1 is: ");
		scanf("%lf", &p1);
	}

	/* Reading the PGM images,
	   allocating memory for the histogram.
	---------------------------------------*/

	if ((imageA = readPGM(nameImageA, &widthA, &heightA)) == NULL)
	{
		printf("\nERROR image \"%s\"\n\n", nameImageA); exit(1);
	}
	if ((imageB = readPGM(nameImageB, &widthB, &heightB)) == NULL)
	{
		printf("\nERROR image \"%s\"\n\n", nameImageB); exit(1);
	}
	if (widthA != widthB || heightA != heightB)
	{
		printf("\nERROR size images\n\n"); exit(1);
	}
	histogram = (double*)malloc((numberOfDirections + 1) * sizeof(double));

	//init phidesc
	/*for (int i=0;i<NBDesc; i++)
		for (int j=0;j<NBMaxDir; j++)
			PhiDesc[i][j]=0;
	*/

	/* Calling the appropriate function, with the appropriate arguments.
	--------------------------------------------------------------------*/

	switch (choice) {
	case 1: FRHistogram_CrispRaster(histogram, numberOfDirections, typeForce,
		imageA, imageB, widthA, heightA);
		break;
	case 2: FRHistogram_FuzzyRaster(histogram, numberOfDirections, typeForce,
		imageA, imageB, widthA, heightA,
		fuzzyMethod);
		break;
	case 3: F02Histogram_CrispRaster(histogram, numberOfDirections,
		imageA, imageB, widthA, heightA, p0, p1);
		break;
	case 4: F02Histogram_FuzzyRaster(histogram, numberOfDirections,
		imageA, imageB, widthA, heightA, p0, p1,
		fuzzyMethod);
	case 5:
		for (typeForce = rdeb; typeForce <= rfin; typeForce += rpas)
		{
			for (i = 0; i <= numberOfDirections; histogram[i++] = 0); // do while pour éviter une de plus...
			FRHistogram_CrispRaster(histogram, numberOfDirections, typeForce, imageA, imageB, widthA, heightA);
			fprintf(histogramFile, "Force %lf ", typeForce);
			fprintf(histogramFile, "NBDIR %d\n", numberOfDirections);
			for (i = 0; i <= numberOfDirections; i++)
				fprintf(histogramFile, "%f ", histogram[i]);
			fprintf(histogramFile, "\n");
		}
		break;
	}

	/* Writing the histogram file.
	------------------------------*/

	/* if((histogramFile=fopen(nameHistogramFile,"wt"))==NULL)
		{printf("\nERROR histogram file\n\n"); exit(1);}
	fprintf(histogramFile,"%d\n",numberOfDirections);
	for(i=0;i<=numberOfDirections;i++)
		fprintf(histogramFile,"%f\n",histogram[i]);*/
	fclose(histogramFile);

	/* Done.
	--------*/
	free(histogram);
	free(imageA);
	free(imageB);
	return 0;
}