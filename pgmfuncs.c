#include <stdio.h>
#include <stdlib.h>


unsigned int pow(char a);
int AsciiToInt(char * asc);

int ReadPGMData (char *fn, unsigned char *target, int width, int height, int dataoffset)
{
	FILE *pgm;
	int rv;

	#pragma warning (disable : 4996)
	pgm = fopen(fn, "rb");

	printf("Reading %s data ...\n", fn);

	fseek (pgm, dataoffset, SEEK_SET);
	printf("(Skipped header)\n", fn);

	if(fread((unsigned char*)target, sizeof(unsigned char), width*height, pgm) == width*height) 
	{
		printf("ok\n");
		rv = 0;
	}
	else
	{
		printf("read failed\n");
		rv = 1;
	}

	fclose(pgm);
	return rv;
}

int ReadPGMHeader(char *fn, int *w, int *h)
{
	FILE *pgm;
	char decimal[5] = {0,0,0,0,0};
	unsigned int phase = 0, i, hl=0;
	int finished = 0;
	unsigned char c, d;

	pgm = fopen(fn, "rb");

	if(!pgm)
	{
		printf("\nReadPGMHeader:open failed");
		printf("\n  for %s\n", fn);
		exit(0);
	}

	while(!finished)
	{
		c = fgetc(pgm);
		hl++;
		while(c == 10)
		{
			c = fgetc(pgm);
			hl++;
		}
		if(c == '#')		// skip comments anywhere
		{
			d = 0;
			while(d != 10)
			{
				d = fgetc(pgm);
				hl++;
			}
		}
		else
		{
			if(phase == 0)	// look for initial 'P'
			{
				if(c == 'P')
					phase = 1;
				else
				{
					printf("\ndid not find P\n");
					return 0;
				}
			}
			else if(phase == 1)	// look for '5'
			{
				if(c == '5')
				{
					phase = 2;
					fgetc(pgm);
					hl++;
				}
				else
				{
					printf("\ndid not find (P)5\n");
					return 0;
				}
			}
			else if(phase == 2)	// read image width
			{
				i = 0;
				d = c;
				while(d >= 48 && d <= 57)
				{
					decimal[i] = d;
					d = fgetc(pgm);
					hl++;
					i++;
				}	
				*(w) = AsciiToInt(decimal);

				if(*(w) < 1 || *(w) > 5000)
				{
					printf("\nInvalid image width:%i\n", *(w));
					return 0;
				}
				else
					phase = 3;
			}
			else if(phase == 3)	// read image height
			{
				d = c;
				for(i = 0; i < 5; i++)
					decimal[i] = 0;
				i = 0;

				while(d >= 48 && d <= 57)
				{
					decimal[i] = d;
					d = fgetc(pgm);
					hl++;
					i++;
				}	
				*(h) = AsciiToInt(decimal);

				if(*(h) < 1 || *(h) > 5000)
				{
					printf("\nInvalid image height:%i\n", *(h));
					return 0;
				}
				else
					phase = 4;
			}
			else if(phase == 4)	// skip maximum brightness
			{
				d = c;
				while(d >= 48 && d <= 57)
				{
					d = fgetc(pgm);
					hl++;
				}
				finished = 1;
			}
		}
	}

	fclose(pgm);

	printf("ok\n");

	return hl;
}

int AsciiToInt(char *asc)
{
	char c;
	int result = 0, i;
	int figcount = 0;

	for(i = 0; i < 5; i++)
		if(asc[i] <= 57 && asc[i] >= 48)
			figcount++;

	for(i = 0; i < figcount; i++)
	{
		c = asc[i];
		if (c > 57 || c < 48)
			break;
		result += (c - 48) * ((unsigned int) pow((figcount-1)-i));
	}
	return result;
} 

unsigned int pow(char a)
{
	switch(a)
	{
	case 4: 
		return 10000;
	break;   
	case 3:
		return 1000;    
	break;
	case 2:
		return 100;
	break;
	case 1:
		return 10;
	}
	return 1;
}

void WriteRAW (char *fn, unsigned char *target, unsigned long size)
{
	FILE *pcx;
	unsigned long i;

	pcx = fopen(fn, "wb");

	for(i = 0; i < size; i++)
		fputc(target[i], pcx);

	fclose(pcx);
}

int WritePGM (char *fn, unsigned char *source, unsigned int width, unsigned int height)
{
	const char c255[] = {50, 53, 53, 10};
	const char cP5[] = {80, 53, 10};
	FILE *pgm;
	int num;
	char txt[5];

	if(width > 9999 || height > 9999)
		return 1;

	pgm = fopen(fn, "wb");

	fwrite(cP5, sizeof(unsigned char), 3, pgm); 	// write P5 identifier and 'NL'

	num = sprintf(txt, "%d", width);
	fwrite(txt, sizeof(unsigned char), num, pgm); 	// write width and whitespace
	fputc(32, pgm);

	num = sprintf(txt, "%d", height);
	fwrite(txt, sizeof(unsigned char), num, pgm); 	// write height and 'NL'
	fputc(10, pgm);

	fwrite(c255, sizeof(unsigned char), 4, pgm); 	// write 255 and 'NL' (255 is maximum intensity)

	fwrite((unsigned char *) source, sizeof(unsigned char), width*height, pgm); 	// write image data

	fclose(pgm);

	return 0;
}
/*
int main(int argc, char **argv)
{
	unsigned char *imgdata = 0;
	int imgsize = 0, w, h, headerlength;

	headerlength = ReadPGMHeader(argv[1], &w, &h);

	imgdata = (unsigned char *) malloc(w*h);

	if(imgdata != 0)
		printf("reserved %i bytes of memory\n", w*h);
	else
	{	
		printf("memory allocation failed\n");
		return 0;
	}

	ReadPGMData(argv[1], imgdata, w, h, headerlength);
	imgsize = w*h;
	if(imgsize)
	{
		printf("writing %i bytes...\n", imgsize);
		WritePGM(argv[2], imgdata, w, h);
		printf("done\n");
	}

	free(imgdata);
}*/
