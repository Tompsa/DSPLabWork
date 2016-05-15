__kernel void depth_estimator_simple (__global unsigned char* imgL,
									  __global unsigned char* imgR,
									  __read_only int width,
									  __read_only int height,
									  __read_only int WIN_SIZE,
									  __read_only int MAX_DISP,
									  __global unsigned char* resultL,
                                      __local unsigned char* local_image,
                                      __local unsigned char* local_image_r, 
                                      __read_only int local_padded_width,
                                      __read_only int local_padded_height,
                                      __read_only int local_padded_width_r)
                                      {
    
    int groupStartCol = get_group_id(0)*get_local_size(0);
    int groupStartRow = get_group_id(1)*get_local_size(1);
    
    int localCol = get_local_id(0);
    int localRow = get_local_id(1);
    
    int X = groupStartCol + localCol; // get_global_id(0);
	int Y = groupStartRow + localRow; //get_global_id(1);

    for(int i = localRow; i < local_padded_height; i += get_local_size(1)){
      int curRow = groupStartRow + i;
          
      for(int j = localCol; j < local_padded_width; j += get_local_size(0)){
          int curCol = groupStartCol + j; 
          if(curRow < height && curCol < width)
            local_image[i*local_padded_width + j] = imgL[curRow * width + curCol]; 
          else
            local_image[i*local_padded_width + j] = 0;
               
      }
      
      for(int j = localCol; j < local_padded_width_r; j += get_local_size(0)){
          int curCol = groupStartCol + j; 
          if(curRow < height && curCol < width + MAX_DISP && curCol - MAX_DISP > 0){
              local_image_r[i*local_padded_width_r + j] = imgR[curRow * width + curCol - MAX_DISP]; 
         }
         else
           local_image_r[i*local_padded_width_r + j]  = 0;  
      }
          
    }
        
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
   if(Y < height && X < width){
	
	int imgDiff = 0;
	int imgDiffSquared = 0;
	int ssd = 0;
	int minSSD = 1000000000; //pow(2,31)-1;
	int bestDisparity = 0;
    
    for (int D = 0; D < MAX_DISP; D++) {
        ssd = 0;
		for (int WIN_Y = 0; WIN_Y < WIN_SIZE; WIN_Y++) {
			for (int WIN_X = 0; WIN_X < WIN_SIZE; WIN_X++) {
                if(localCol + groupStartCol + WIN_X < width && localRow + groupStartRow + WIN_Y < height){
				  imgDiff = local_image[(localRow+WIN_Y)*local_padded_width + localCol + WIN_X] - local_image_r[(localRow+WIN_Y)*local_padded_width_r + localCol + WIN_X + MAX_DISP - D];
				  imgDiffSquared = imgDiff*imgDiff;
				  ssd += imgDiffSquared;
                }
			}
		}
		if (ssd < minSSD) {
			minSSD = ssd;
			bestDisparity = D;
		}
	}
    
    resultL[(X)  + (Y)*width] = (unsigned char) (bestDisparity*255 / MAX_DISP);
   }
}


__kernel void depth_estimator_simple_r (__global unsigned char* imgL,
									  __global unsigned char* imgR,
									  __read_only int width,
									  __read_only int height,
									  __read_only int WIN_SIZE,
									  __read_only int MAX_DISP,
									  __global unsigned char* resultL,
                                      __local unsigned char* local_image_r,
                                      __local unsigned char* local_image, 
                                      __read_only int local_padded_width_r,
                                      __read_only int local_padded_height,
                                      __read_only int local_padded_width)
                                      {
						
    int groupStartCol = get_group_id(0)*get_local_size(0);
    int groupStartRow = get_group_id(1)*get_local_size(1);
    
    int localCol = get_local_id(0);
    int localRow = get_local_id(1);
    
    int X = groupStartCol + localCol; // get_global_id(0);
	int Y = groupStartRow + localRow; //get_global_id(1);
    
    for(int i = localRow; i < local_padded_height; i += get_local_size(1)){
      int curRow = groupStartRow + i;
          
      for(int j = localCol; j < local_padded_width; j += get_local_size(0)){
          int curCol = groupStartCol + j; 
          if(curRow < height && curCol < width)
            local_image[i*local_padded_width + j] = imgL[curRow * width + curCol]; 
          else
            local_image[i*local_padded_width + j] = 0;
               
      }
      
      for(int j = localCol; j < local_padded_width_r; j += get_local_size(0)){
          int curCol = groupStartCol + j; 
          if(curRow < height && curCol < width)
              local_image_r[i*local_padded_width_r + j] = imgR[curRow * width + curCol]; 
          else
             local_image_r[i*local_padded_width_r + j]  = 0;  
      }
          
    }
        
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
   if(Y < height && X < width){
	
	int imgDiff = 0;
	int imgDiffSquared = 0;
	int ssd = 0;
	int minSSD = 1000000000; //pow(2,31)-1;
	int bestDisparity = 0;
    
    for (int D = 0; D < MAX_DISP; D++) {
        ssd = 0;
		for (int WIN_Y = 0; WIN_Y < WIN_SIZE; WIN_Y++) {
			for (int WIN_X = 0; WIN_X < WIN_SIZE; WIN_X++) {
                if(localCol + groupStartCol + WIN_X < width && localRow + groupStartRow + WIN_Y < height){
				  imgDiff = local_image[(localRow+WIN_Y)*local_padded_width + localCol + WIN_X + D] - local_image_r[(localRow+WIN_Y)*local_padded_width_r + localCol + WIN_X];
				  imgDiffSquared = imgDiff*imgDiff;
				  ssd += imgDiffSquared;
                }
			}
		}
		if (ssd < minSSD) {
			minSSD = ssd;
			bestDisparity = D;
		}
	}
    
    resultL[(X)  + (Y)*width] = (unsigned char) (bestDisparity*255 / MAX_DISP);
   }
}


__kernel void cross_check (__global unsigned char* resultL,
						   __global unsigned char* resultR,
						   __read_only int threshold ) {	
						                          
    const int X = get_global_id(0);
	const int Y = get_global_id(1);
    //resultL[X + Y*427] = resultR[X + Y*427-(resultL[X + Y*427])*90/255];
    
    //unsigned char disp = resultL[X + Y*427];   //(unsigned char) (resultL[pos1]*90 / 255);;
	if ((resultL[X + Y*427] - resultR[X + Y*427-(resultL[X + Y*427])*90/255]) > threshold) {
		resultL[X + Y*427] = 0;
	}
}


__kernel void occlusion_filling (__global unsigned char* resultL,
									__read_only int width, 
                                    __read_only int height) {
	
	const int X = get_global_id(0);
    const int Y = get_global_id(1);
	
	if (resultL[X+Y*width] == 0) {
		int max = 0;
		int c = 1;
		for(int i =-14; i < 15; i++){
			for(int j = -14; j < 15; j++){
				//int y = width * j;
				if(resultL[X+i+(j+Y)*width] != 0 && X+i+(j+Y)*width > 0 && X+i+(j+Y)*width < height*width) {
					max += resultL[X+i+(j+Y)*width];
					c++;
				}
                
			}
		}

		resultL[X+Y*width] = max / c;
	}
  
}