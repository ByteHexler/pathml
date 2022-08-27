  def extractAnnotationTiles(self, outputDir, slideName=False, numTilesToExtractPerClass='all', classesToExtract=False, otherClassNames=False,
      extractSegmentationMasks=False, tileAnnotationOverlapThreshold=0.5, foregroundLevelThreshold=False, otsuLevelThreshold=False, triangleLevelThreshold=False, tissueLevelThreshold=False,
      returnTileStats=True, returnOnlyNumTilesFromThisClass=False, seed=False):
      """A function to extract tiles that overlap with annotations into
      directory structure amenable to torch.utils.data.ConcatDataset.

      Args:
          outputDir (str): the path to the directory where the tile directory will be stored
          slideName (str, optional): the name of the slide to be used in the file names of the extracted tiles and masks. Default is Slide.slideFileName.
          numTilesToExtractPerClass (dict or int or 'all', optional): how many suitable tiles to extract from the slide for each class; if more suitable tiles are available than are requested, tiles will be chosen at random; expected to be positive integer, a dictionary with class names as keys and positive integers as values, or 'all' to extract all suitable tiles for each class. Default is 'all'.
          classesToExtract (str or list of str, optional): defaults to extracting all classes found in the annotations, but if defined, must be a string or a list of strings of class names.
          otherClassNames (str or list of str, optional): if defined, creates an empty class directory alongside the unannotated class directory for each class name in the list (or string) for torch ImageFolder purposes. If set to 'discernFromClassesToExtract', empty class directories will be created for all classes not found in annotations. Default is False.
          extractSegmentationMasks (Bool, optional): whether to extract a 'masks' directory that is exactly parallel to the 'tiles' directory, and contains binary segmentation mask tiles for each class desired. Pixel values of 255 in these masks appear as white and indicate the presence of the class; pixel values of 0 appear as black and indicate the absence of the class. Default is False.
          tileAnnotationOverlapThreshold (float, optional): a number greater than 0 and less than or equal to 1, or a dictionary of such values, with a key for each class to extract. The numbers specify the minimum fraction of a tile's area that overlaps a given class's annotations for it to be extracted. Default is 0.5.
          foregroundLevelThreshold (str or int or float, optional): if defined as an int, only extracts tiles with a 0-100 foregroundLevel value less or equal to than the set value (0 is a black tile, 100 is a white tile). Only includes Otsu's method-passing tiles if set to 'otsu', or triangle algorithm-passing tiles if set to 'triangle'. Default is not to filter on foreground at all.
          tissueLevelThreshold (Bool, optional): if defined, only extracts tiles with a 0 to 1 tissueLevel probability greater than or equal to the set value. Default is False.
          returnTileStats (Bool, optional): whether to return the 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation. Default is True.
          returnOnlyNumTilesFromThisClass (str, optional): causes only the number of suitable tiles for the specified class in the slide; no tile images are created if a string is provided. Default is False.
          seed (int, optional): the random seed to use for reproducible anayses. Default is not to use a seed when randomly selecting tiles.

      Returns:
          dict: A dictionary containing the Slide's name, 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation; if returnTileStats is set to False, True will be returned

      Example:
          channel_data = pathml_slide.extractAnnotationTiles('/path/to/directory', numTilesToExtractPerClass=200, tissueLevelThreshold=0.995)
      """

      if not self.hasTileDictionary():
          raise PermissionError(
              'setTileProperties must be called before extracting tiles')
      if not self.hasAnnotations():
          raise PermissionError(
              'addAnnotations must be called before extracting tiles')
      if seed:
          if type(seed) != int:
              raise ValueError('Seed must be an integer')
          random.seed(seed)
      # get case ID
      if slideName:
          if type(slideName) != str:
              raise ValueError("slideName must be a string")
          else:
              id = slideName
      else:
          id = self.slideFileId

      # get classes to extract
      extractionClasses = []
      if otherClassNames == 'discernFromClassesToExtract':
          extraClasses = []
      if not classesToExtract:
          for key, value in self.tileDictionary[list(self.tileDictionary.keys())[0]].items():
              if 'Overlap' in key:
                  extractionClasses.append(key)
      elif (type(classesToExtract) == list) or (type(classesToExtract) == str):
          if type(classesToExtract) == str:
              classesToExtract = [classesToExtract]
          for classToExtract in classesToExtract:
              extractionClass = classToExtract+'Overlap'
              if extractionClass not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                  if otherClassNames == 'discernFromClassesToExtract':
                      extraClasses.append(extractionClass)
                  else:
                      raise ValueError(extractionClass+' not found in tile dictionary')
              else:
                  extractionClasses.append(extractionClass)
      else:
          raise ValueError("classesToExtract must be a string or list of strings")

      extractionClasses = [extractionClass.split('Overlap')[0] for extractionClass in extractionClasses]
      print('Found '+str(len(extractionClasses))+' class(es) to extract in annotations:', extractionClasses)

      if otherClassNames == 'discernFromClassesToExtract':
          extraClasses = [extraClass.split('Overlap')[0] for extraClass in extraClasses]
          print('Found '+str(len(extraClasses))+' class(es) to extract in not present in annotations:', extraClasses)

      # Convert annotationOverlapThreshold into a dictionary (if necessary)
      annotationOverlapThresholdDict = {}
      if (type(tileAnnotationOverlapThreshold) == int) or (type(tileAnnotationOverlapThreshold) == float):
          if (tileAnnotationOverlapThreshold <= 0) or (tileAnnotationOverlapThreshold > 1):
              raise ValueError('tileAnnotationOverlapThreshold must be greater than 0 and less than or equal to 1')
          for extractionClass in extractionClasses:
              annotationOverlapThresholdDict[extractionClass] = tileAnnotationOverlapThreshold
      elif type(tileAnnotationOverlapThreshold) == dict:
          for ec, taot in tileAnnotationOverlapThreshold.items():
              if ec not in extractionClasses:
                  raise ValueError('Class '+str(ec)+' present as a key in tileAnnotationOverlapThreshold but absent from the tileDictionary')
              if ((type(taot) != int) and (type(taot) != float)) or ((taot <= 0) or (taot > 1)):
                  raise ValueError('Tile annotation overlap threshold of class '+str(ec)+' must be a number greater than zero and less than or equal to 1')
          for extractionClass in extractionClasses:
              if extractionClass not in tileAnnotationOverlapThreshold:
                  raise ValueError('Class '+str(extractionClass)+' present in the tileDictionary but not present as a key in tileAnnotationOverlapThreshold')
          annotationOverlapThresholdDict = tileAnnotationOverlapThreshold
      else:
          raise ValueError('tileAnnotationOverlapThreshold must be a dictionary or number greater than 0 and less than or equal to 1')

      if tissueLevelThreshold:
          if ((type(tissueLevelThreshold) != int) and (type(tissueLevelThreshold) != float)) or ((tissueLevelThreshold <= 0) or (tissueLevelThreshold > 1)):
              raise ValueError('tissueLevelThreshold must be a number greater than zero and less than or equal to 1')

      # Get tiles to extract
      annotatedTileAddresses = {extractionClass: [] for extractionClass in extractionClasses}

      suitable_tile_addresses = self.suitableTileAddresses(tissueLevelThreshold=tissueLevelThreshold, foregroundLevelThreshold=foregroundLevelThreshold, otsuLevelThreshold=otsuLevelThreshold, triangleLevelThreshold=triangleLevelThreshold)
      for address in suitable_tile_addresses:
          for extractionClass in extractionClasses:
              if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
                  annotatedTileAddresses[extractionClass].append(address)

      annotatedTilesToExtract = {} #{extractionClass: [] for extractionClass in extractionClasses}
      if type(numTilesToExtractPerClass) == int:
          if numTilesToExtractPerClass <= 0:
              raise ValueError('If numTilesToExtractPerClass is an integer, it must be greater than 0')
          for extractionClass in extractionClasses:
              if len(annotatedTileAddresses[extractionClass]) == 0:
                  print('Warning: 0 suitable '+extractionClass+' tiles found')
              if len(annotatedTileAddresses[extractionClass]) < numTilesToExtractPerClass:
                  print('Warning: '+str(len(annotatedTileAddresses[extractionClass]))+' suitable '+extractionClass+' tiles found but requested '+str(numTilesToExtractPerClass)+' tiles to extract. Extracting all suitable tiles...')
                  annotatedTilesToExtract[extractionClass] = annotatedTileAddresses[extractionClass]
              else:
                  annotatedTilesToExtract[extractionClass] = random.sample(annotatedTileAddresses[extractionClass], numTilesToExtractPerClass)

      elif numTilesToExtractPerClass == 'all':
          for extractionClass in extractionClasses:
              if len(annotatedTileAddresses[extractionClass]) == 0:
                  print('Warning: 0 suitable '+extractionClass+' tiles found')
              if len(annotatedTileAddresses[extractionClass]) > 500:
                  print('Warning: '+str(len(annotatedTileAddresses[extractionClass]))+' suitable '+extractionClass+' tiles found')
              annotatedTilesToExtract[extractionClass] = annotatedTileAddresses[extractionClass]

      elif type(numTilesToExtractPerClass) == dict:
          for ec,tc in numTilesToExtractPerClass.items():
              if ec not in extractionClasses:
                  raise Warning('Class '+ec+' present as a key in numTilesToExtractPerClass dictionary but absent from the tileDictionary')
          for extractionClass in extractionClasses:
              if len(annotatedTileAddresses[extractionClass]) == 0:
                  print('Warning: 0 suitable '+extractionClass+' tiles found')
              if extractionClass not in numTilesToExtractPerClass:
                  raise ValueError(extractionClass+' not present in the numTilesToExtractPerClass dictionary')
              numTiles = numTilesToExtractPerClass[extractionClass]
              if (type(numTiles) != int) or (numTiles <= 0):
                  raise ValueError(extractionClass+' does not have a positive integer set as its value in the numTilesToExtractPerClass dictionary')
              if len(annotatedTileAddresses[extractionClass]) < numTiles:
                  print('Warning: '+str(len(annotatedTileAddresses[extractionClass]))+' suitable '+extractionClass+' tiles found but requested '+str(numTiles)+' tiles to extract. Extracting all suitable tiles...')
                  annotatedTilesToExtract[extractionClass] = annotatedTileAddresses[extractionClass]
              else:
                  annotatedTilesToExtract[extractionClass] = random.sample(annotatedTileAddresses[extractionClass], numTiles)

      else:
          raise ValueError("numTilesToExtractPerClass must be a positive integer, a dictionary, or 'all'")

      # Create empty class tile directories for extractionClasses with at least one suitable tile
      if not returnOnlyNumTilesFromThisClass:
          for extractionClass,tte in annotatedTilesToExtract.items():
              if len(tte) > 0:
                  try:
                      os.makedirs(os.path.join(outputDir, 'tiles', id, extractionClass), exist_ok=True)
                  except:
                      raise ValueError(os.path.join(outputDir, 'tiles', id, extractionClass)+' is not a valid path')
                  if otherClassNames:
                      if otherClassNames == 'discernFromClassesToExtract':
                          for extraClass in extraClasses:
                              if type(extraClass) != str:
                                  raise ValueError('Class to extract not found in annotations '+str(extraClass)+' is not a string.')
                              try:
                                  os.makedirs(os.path.join(outputDir, 'tiles', id, extraClass), exist_ok=True)
                              except:
                                  raise ValueError(os.path.join(outputDir, 'tiles', id, extraClass)+' is not a valid path')
                      elif type(otherClassNames) == str:
                          try:
                              os.makedirs(os.path.join(outputDir, 'tiles', id, otherClassNames), exist_ok=True)
                          except:
                              raise ValueError(os.path.join(outputDir, 'tiles', id, otherClassNames)+' is not a valid path')
                      elif type(otherClassNames) == list:
                          for otherClassName in otherClassNames:
                              if type(otherClassName) != str:
                                  raise ValueError('If otherClassNames is a list, all elements of list must be strings')
                              try:
                                  os.makedirs(os.path.join(outputDir, 'tiles', id, otherClassName), exist_ok=True)
                              except:
                                  raise ValueError(os.path.join(outputDir, 'tiles', id, otherClassName)+' is not a valid path')
                      else:
                          raise ValueError('otherClassNames must be a string or list of strings')

                  # Create empty class mask directory (if desired)
                  if extractSegmentationMasks:
                      try:
                          os.makedirs(os.path.join(outputDir, 'masks', id, extractionClass), exist_ok=True)
                      except:
                          raise ValueError(os.path.join(outputDir, 'masks', id, extractionClass)+' is not a valid path')
                      if otherClassNames:
                          if otherClassNames == 'discernFromClassesToExtract':
                              for extraClass in extraClasses:
                                  if type(extraClass) != str:
                                      raise ValueError('Class to extract not found in annotations '+str(extraClass)+' is not a string.')
                                  try:
                                      os.makedirs(os.path.join(outputDir, 'masks', id, extraClass), exist_ok=True)
                                  except:
                                      raise ValueError(os.path.join(outputDir, 'masks', id, extraClass)+' is not a valid path')
                          elif type(otherClassNames) == str:
                              try:
                                  os.makedirs(os.path.join(outputDir, 'masks', id, otherClassNames), exist_ok=True)
                              except:
                                  raise ValueError(os.path.join(outputDir, 'masks', id, otherClassNames)+' is not a valid path')
                          elif type(otherClassNames) == list:
                              for otherClassName in otherClassNames:
                                  if type(otherClassName) != str:
                                      raise ValueError('If otherClassNames is a list, all elements of list must be strings')
                                  try:
                                      os.makedirs(os.path.join(outputDir, 'masks', id, otherClassName), exist_ok=True)
                                  except:
                                      raise ValueError(os.path.join(outputDir, 'masks', id, otherClassName)+' is not a valid path')
                          else:
                              raise ValueError('otherClassNames must be a string or list of strings')

      channel_sums = np.zeros(3)
      channel_squared_sums = np.zeros(3)
      tileCounter = 0
      normalize_to_1max = transforms.Compose([transforms.ToTensor()])

      # Extract tiles
      for ec,tte in annotatedTilesToExtract.items():
          if returnOnlyNumTilesFromThisClass and ec == returnOnlyNumTilesFromThisClass:
              return(len(annotatedTileAddresses[ec]))

          if len(tte) > 0:
              if extractSegmentationMasks:
                  print("Extracting "+str(len(tte))+" of "+str(len(annotatedTileAddresses[ec]))+" "+ec+" tiles and segmentation masks...")
              else:
                  print("Extracting "+str(len(tte))+" of "+str(len(annotatedTileAddresses[ec]))+" "+ec+" tiles...")

          for tl in tte:
              area = self.getTile(tl)
              if (tissueLevelThreshold) and (foregroundLevelThreshold):
                  area.write_to_file(os.path.join(outputDir, 'tiles', id, ec,
                      id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
              elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
                  area.write_to_file(os.path.join(outputDir, 'tiles', id, ec,
                      id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel.jpg'), Q=100)
              elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
                  area.write_to_file(os.path.join(outputDir, 'tiles', id, ec,
                      id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
              else:
                  area.write_to_file(os.path.join(outputDir, 'tiles', id, ec,
                      id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize.jpg'), Q=100)

              tileCounter = tileCounter + 1
              if returnTileStats:
                  nparea = self.getTile(tl, writeToNumpy=True)[...,:3] # remove transparency channel
                  nparea = normalize_to_1max(nparea).numpy() # normalize values from 0-255 to 0-1
                  local_channel_sums = np.sum(nparea, axis=(1,2))
                  local_channel_squared_sums = np.sum(np.square(nparea), axis=(1,2))
                  channel_sums = np.add(channel_sums, local_channel_sums)
                  channel_squared_sums = np.add(channel_squared_sums, local_channel_squared_sums)

              # Extract segmentation masks
              if extractSegmentationMasks:
                  mask = self.getAnnotationTileMask(tl, ec)
                  if (tissueLevelThreshold) and (foregroundLevelThreshold):
                      mask.save(os.path.join(outputDir, 'masks', id, ec,
                          id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'))
                  elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
                      mask.save(os.path.join(outputDir, 'masks', id, ec,
                          id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_mask.gif'))
                  elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
                      mask.save(os.path.join(outputDir, 'masks', id, ec,
                          id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'))
                  else:
                      mask.save(os.path.join(outputDir, 'masks', id, ec,
                          id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_mask.gif'))

      if returnOnlyNumTilesFromThisClass:
          raise Warning(returnOnlyNumTilesFromThisClass+' not found in tile dictionary')

      if tileCounter == 0:
          print('Warning: 0 suitable annotated tiles found across all classes; making no tile directories and returning zeroes')

      if returnTileStats:
          if slideName:
              return {'slide': slideName,
                      'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                      'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                      'num_tiles': tileCounter}
          else:
              return {'slide': self.slideFileName,
                      'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                      'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                      'num_tiles': tileCounter}
      else:
          return True


  def extractAnnotationTilesMultiClassSegmentation(self, outputDir, slideName=False, numTilesToExtract=100, classesToExtract=False,
      tileAnnotationOverlapThreshold=0.5, foregroundLevelThreshold=False, otsuLevelThreshold=False, triangleLevelThreshold=False, tissueLevelThreshold=False, returnTileStats=True, seed=False):
      """A function to extract tiles that overlap with annotations and their
      corresponding segmentation masks, where annotation masks are returned as
      .npy files containing ndarray stacks (each array in the stack being one
      class's segmentation class) for use in multi-class segmentation problems.

      Args:
          outputDir (str): the path to the directory where the tile directory will be stored
          slideName (str, optional): the name of the slide to be used in the file names of the extracted tiles and masks. Default is Slide.slideFileName.
          numTilesToExtractPerClass (int or 'all', optional): how many suitable tiles to extract from the slide; if more suitable tiles are available than are requested, tiles will be chosen at random; expected to be positive integer or 'all' to extract all suitable tiles for each class. Default is 'all'.
          classesToExtract (str or list of str, optional): which classes to consider when selecting tiles and making mask stacks; defaults to extracting all classes found in the annotations, but if defined, must be a string or a list of strings of class names.
          tileAnnotationOverlapThreshold (float, optional): a number greater than 0 and less than or equal to 1, or a dictionary of such values, with a key for each class to extract. The numbers specify the minimum fraction of a tile's area that overlaps the annotations of the classesToExtract for that tile to be extracted. The overlaps with all classesToExtract classes are summed together and if this sum is greater or equal to tileAnnotationOverlapThreshold, then the tile is extracted. Default is 0.5.
          foregroundLevelThreshold (str or int or float, optional): if defined as an int, only extracts tiles with a 0-100 foregroundLevel value less or equal to than the set value (0 is a black tile, 100 is a white tile). Only includes Otsu's method-passing tiles if set to 'otsu', or triangle algorithm-passing tiles if set to 'triangle'. Default is not to filter on foreground at all.
          tissueLevelThreshold (Bool, optional): if defined, only extracts tiles with a 0 to 1 tissueLevel probability greater than or equal to the set value. Default is False.
          returnTileStats (Bool, optional): whether to return the 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation. Default is True.
          seed (int, optional): the random seed to use for reproducible anayses. Default is not to use a seed when randomly selecting tiles.

      Returns:
          dict: A dictionary containing the class order that the class masks appear in ndarray mask stacks, the Slide's name, 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation; if returnTileStats is set to False, only the class mask order will be returned (a list of strings)

      Example:
          channel_data = pathml_slide.extractAnnotationTilesMultiClassSegmentation('/path/to/directory', numTilesToExtractPerClass=200, classesToExtract=['lymphocyte', 'normal', 'tumor'], tileAnnotationOverlapThreshold=0.6, tissueLevelThreshold=0.995)
      """

      if not self.hasTileDictionary():
          raise PermissionError(
              'setTileProperties must be called before extracting tiles')
      if not self.hasAnnotations():
          raise PermissionError(
              'addAnnotations must be called before extracting tiles')
      if seed:
          if type(seed) != int:
              raise ValueError('Seed must be an integer')
          random.seed(seed)
      # get case ID
      if slideName:
          if type(slideName) != str:
              raise ValueError("slideName must be a string")
          else:
              id = slideName
      else:
          id = self.slideFileId

      # get classes to extract
      extractionClasses = []
      if not classesToExtract:
          for key, value in self.tileDictionary[list(self.tileDictionary.keys())[0]].items():
              if 'Overlap' in key:
                  extractionClasses.append(key)
      elif (type(classesToExtract) == list) or (type(classesToExtract) == str):
          if type(classesToExtract) == str:
              classesToExtract = [classesToExtract]
          for classToExtract in classesToExtract:
              extractionClass = classToExtract+'Overlap'
              if extractionClass not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                  raise ValueError(extractionClass+' not found in tile dictionary')
              else:
                  extractionClasses.append(extractionClass)
      else:
          raise ValueError("classesToExtract must be a string or list of strings")

      extractionClasses = [extractionClass.split('Overlap')[0] for extractionClass in extractionClasses]
      print('Found '+str(len(extractionClasses))+' class(es) to extract in annotations:', extractionClasses)

      # Convert annotationOverlapThreshold into a dictionary (if necessary)
      if (type(tileAnnotationOverlapThreshold) == int) or (type(tileAnnotationOverlapThreshold) == float):
          if (tileAnnotationOverlapThreshold <= 0) or (tileAnnotationOverlapThreshold > 1):
              raise ValueError('tileAnnotationOverlapThreshold must be greater than 0 and less than or equal to 1')
      else:
          raise ValueError('tileAnnotationOverlapThreshold must be an int or float greater than 0 and less than or equal to 1')

      if tissueLevelThreshold:
          if ((type(tissueLevelThreshold) != int) and (type(tissueLevelThreshold) != float)) or ((tissueLevelThreshold <= 0) or (tissueLevelThreshold > 1)):
              raise ValueError('tissueLevelThreshold must be a number greater than zero and less than or equal to 1')

      # Get tiles to extract
      annotatedTileAddresses = [] #{extractionClass: [] for extractionClass in extractionClasses}

      # Find tile addresses where the sum of the overlap of the desired classes is at least as big as tileAnnotationOverlapThreshold
      suitable_tile_addresses = self.suitableTileAddresses(tissueLevelThreshold=tissueLevelThreshold, foregroundLevelThreshold=foregroundLevelThreshold, otsuLevelThreshold=otsuLevelThreshold, triangleLevelThreshold=triangleLevelThreshold)
      for address in suitable_tile_addresses:
          classOverlapSum = 0
          for extractionClass in extractionClasses:
              classOverlapSum += self.tileDictionary[address][extractionClass+'Overlap']
          if classOverlapSum >= tileAnnotationOverlapThreshold:
              annotatedTileAddresses.append(address)

      # Choose from among the passing tiles to get the tiles to actually extract
      annotatedTilesToExtract = []#{} #{extractionClass: [] for extractionClass in extractionClasses}
      if type(numTilesToExtract) == int:
          if numTilesToExtract <= 0:
              raise ValueError('If numTilesToExtract is an integer, it must be greater than 0')
          if len(annotatedTileAddresses) == 0:
              print('Warning: 0 suitable tiles found')
          if len(annotatedTileAddresses) < numTilesToExtract:
              print('Warning: '+str(len(annotatedTileAddresses))+' suitable tiles found but requested '+str(numTilesToExtract)+' tiles to extract. Extracting all suitable tiles...')
              annotatedTilesToExtract = annotatedTileAddresses
          else:
              annotatedTilesToExtract = random.sample(annotatedTileAddresses, numTilesToExtract)

      elif numTilesToExtractPerClass == 'all':
          if len(annotatedTileAddresses) == 0:
              print('Warning: 0 suitable tiles found')
          if len(annotatedTileAddresses) > 500:
              print('Warning: '+str(len(annotatedTileAddresses))+' suitable tiles found')
          annotatedTilesToExtract = annotatedTileAddresses

      else:
          raise ValueError("numTilesToExtract must be a positive integer or 'all'")

      # Create empty class tile and mask directories
      try:
          os.makedirs(os.path.join(outputDir, 'tiles', id), exist_ok=True)
      except:
          raise ValueError(os.path.join(outputDir, 'tiles', id)+' is not a valid path')
      try:
          os.makedirs(os.path.join(outputDir, 'masks', id), exist_ok=True)
      except:
          raise ValueError(os.path.join(outputDir, 'masks', id)+' is not a valid path')

      channel_sums = np.zeros(3)
      channel_squared_sums = np.zeros(3)
      tileCounter = 0
      normalize_to_1max = transforms.Compose([transforms.ToTensor()])

      # Extract tiles
      print("Extracting "+str(len(annotatedTilesToExtract))+" tiles and segmentation masks...")
      for tl in annotatedTilesToExtract:
          #for tl in tte:
          area = self.getTile(tl)
          if (tissueLevelThreshold) and (foregroundLevelThreshold):
              area.write_to_file(os.path.join(outputDir, 'tiles', id,
                  id+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
          elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
              area.write_to_file(os.path.join(outputDir, 'tiles', id,
                  id+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel.jpg'), Q=100)
          elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
              area.write_to_file(os.path.join(outputDir, 'tiles', id,
                  id+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
          else:
              area.write_to_file(os.path.join(outputDir, 'tiles', id,
                  id+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize.jpg'), Q=100)

          tileCounter = tileCounter + 1
          if returnTileStats:
              nparea = self.getTile(tl, writeToNumpy=True)[...,:3] # remove transparency channel
              nparea = normalize_to_1max(nparea).numpy() # normalize values from 0-255 to 0-1
              local_channel_sums = np.sum(nparea, axis=(1,2))
              local_channel_squared_sums = np.sum(np.square(nparea), axis=(1,2))
              channel_sums = np.add(channel_sums, local_channel_sums)
              channel_squared_sums = np.add(channel_squared_sums, local_channel_squared_sums)

          # Extract multi-class segmentation masks
          all_class_masks = []
          for ec in extractionClasses:
              all_class_masks.append(self.getAnnotationTileMask(tl, ec, writeToNumpy=True, acceptTilesWithoutClass=True)) # allow blank masks to returned

          # Stack class masks into a 3D numpy ndarray (dimensions are: num. classes, tile pixel height, tile pixel width)
          mask_stack = np.stack(all_class_masks, axis=0)

          # Save mask stack as .npy file
          if (tissueLevelThreshold) and (foregroundLevelThreshold):
              np.save(os.path.join(outputDir, 'masks', id,
                  id+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'),
                  mask_stack)
          elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
              np.save(os.path.join(outputDir, 'masks', id,
                  id+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_mask.gif'),
                  mask_stack)
          elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
              np.save(os.path.join(outputDir, 'masks', id,
                  id+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'),
                  mask_stack)
          else:
              np.save(os.path.join(outputDir, 'masks', id,
                  id+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_mask.gif'),
                  mask_stack)

      if tileCounter == 0:
          print('Warning: 0 suitable annotated tiles found across all classes; making no tile directories and returning zeroes')

      if returnTileStats:
          if slideName:
              return {'class_order_in_mask_stack': extractionClasses,
                      'slide': slideName,
                      'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                      'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                      'num_tiles': tileCounter}
          else:
              return {'class_order_in_mask_stack': extractionClasses,
                      'slide': self.slideFileName,
                      'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                      'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                      'num_tiles': tileCounter}
      else:
          return extractionClasses

  def extractRandomUnannotatedTiles(self, outputDir, slideName=False, numTilesToExtract=100, unannotatedClassName='unannotated', otherClassNames=False,
      extractSegmentationMasks=False, foregroundLevelThreshold=False, otsuLevelThreshold=False, triangleLevelThreshold=False, tissueLevelThreshold=False, returnTileStats=True, seed=False):
      """A function to extract randomly selected tiles that don't overlap any
      annotations into directory structure amenable to torch.utils.data.ConcatDataset

      Args:
          outputDir (str): the path to the directory where the tile directory will be stored
          slideName (str, optional): the name of the slide to be used in the file names of the extracted tiles and masks. Default is Slide.slideFileName.
          numTilesToExtract (int, optional): the number of random unannotated tiles to extract. Default is 50.
          unannotatedClassName (str, optional): the name that the unannotated "class" directory should be called. Default is "unannotated".
          otherClassNames (str or list of str, optional): if defined, creates an empty class directory alongside the unannotated class directory for each class name in the list (or string) for torch ImageFolder purposes
          extractSegmentationMasks (Bool, optional): whether to extract a 'masks' directory that is exactly parallel to the 'tiles' directory, and contains binary segmentation mask tiles for each class desired (these tiles will of course all be entirely black, pixel values of 0). Default is False.
          foregroundLevelThreshold (str or int or float, optional): if defined as an int, only extracts tiles with a 0-100 foregroundLevel value less or equal to than the set value (0 is a black tile, 100 is a white tile). Only includes Otsu's method-passing tiles if set to 'otsu', or triangle algorithm-passing tiles if set to 'triangle'. Default is not to filter on foreground at all.
          tissueLevelThreshold (Bool, optional): if defined, only extracts tiles with a 0 to 1 tissueLevel probability greater than or equal to the set value. Default is False.
          returnTileStats (Bool, optional): whether to return the 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation. Default is True.
          seed (int, optional): the random seed to use for reproducible anayses. Default is not to use a seed when randomly selecting tiles.

      Returns:
          dict: A dictionary containing the Slide's name, 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation; if returnTileStats is set to False, True will be returned

      Example:
          channel_data = pathml_slide.extractRandomUnannotatedTiles('/path/to/directory', numTilesToExtract=200, unannotatedClassName="non_metastasis", tissueLevelThreshold=0.995)
      """

      if not self.hasTileDictionary():
          raise PermissionError(
              'setTileProperties must be called before extracting tiles')

      if seed:
          if type(seed) != int:
              raise ValueError('Seed must be an integer')
          random.seed(seed)

      # get case ID
      if slideName:
          if type(slideName) != str:
              raise ValueError("slideName must be a string")
          else:
              id = slideName
      else:
          id = self.slideFileId

      if tissueLevelThreshold:
          if ((type(tissueLevelThreshold) != int) and (type(tissueLevelThreshold) != float)) or ((tissueLevelThreshold <= 0) or (tissueLevelThreshold > 1)):
              raise ValueError('tissueLevelThreshold must be a number greater than zero and less than or equal to 1')
      if (type(numTilesToExtract) != int) or (numTilesToExtract <= 0):
          raise ValueError('numTilesToExtract must be a integer greater than 0')

      # get classes to NOT extract
      annotationClasses = []
      for key, value in self.tileDictionary[list(self.tileDictionary.keys())[0]].items():
          if 'Overlap' in key:
              annotationClasses.append(key)
      if len(annotationClasses) == 0:
          print('No annotations found in tile dictionary; sampling randomly from all suitable tiles')
          #raise Warning('No annotations currently added to Slide tile dictionary; annotations can be added with addAnnotations()')

      # Collect all unannotated tiles
      unannotatedTileAddresses = []

      suitable_tile_addresses = self.suitableTileAddresses(tissueLevelThreshold=tissueLevelThreshold, foregroundLevelThreshold=foregroundLevelThreshold, otsuLevelThreshold=otsuLevelThreshold, triangleLevelThreshold=triangleLevelThreshold)
      for address in suitable_tile_addresses:
          overlapsAnnotation = False
          for annotationClass in annotationClasses:
              if self.tileDictionary[address][annotationClass] > 0:
                  overlapsAnnotation = True
                  break
          if not overlapsAnnotation:
              unannotatedTileAddresses.append(address)

      if len(unannotatedTileAddresses) == 0:
          print('Warning: 0 unannotated tiles found; making no tile directories and returning zeroes')
      if len(unannotatedTileAddresses) < numTilesToExtract:
          print('Warning: '+str(len(unannotatedTileAddresses))+' unannotated tiles found but requested '+str(numTilesToExtract)+' tiles to extract. Extracting all suitable tiles...')
          unannotatedTilesToExtract = unannotatedTileAddresses
      else:
          unannotatedTilesToExtract = random.sample(unannotatedTileAddresses, numTilesToExtract)

      # Create empty class tile directories
      if len(unannotatedTileAddresses) > 0:
          try:
              os.makedirs(os.path.join(outputDir, 'tiles', id, unannotatedClassName), exist_ok=True)
          except:
              raise ValueError(os.path.join(outputDir, 'tiles', id, unannotatedClassName)+' is not a valid path')
          if otherClassNames:
              if type(otherClassNames) == str:
                  try:
                      os.makedirs(os.path.join(outputDir, 'tiles', id, otherClassNames), exist_ok=True)
                  except:
                      raise ValueError(os.path.join(outputDir, 'tiles', id, otherClassNames)+' is not a valid path')
              elif type(otherClassNames) == list:
                  for otherClassName in otherClassNames:
                      if type(otherClassName) != str:
                          raise ValueError('If otherClassNames is a list, all elements of list must be strings')
                      try:
                          os.makedirs(os.path.join(outputDir, 'tiles', id, otherClassName), exist_ok=True)
                      except:
                          raise ValueError(os.path.join(outputDir, 'tiles', id, otherClassName)+' is not a valid path')
              else:
                  raise ValueError('otherClassNames must be a string or list of strings')

          # Create empty class mask directory (if desired)
          if extractSegmentationMasks:
              try:
                  os.makedirs(os.path.join(outputDir, 'masks', id, unannotatedClassName), exist_ok=True)
              except:
                  raise ValueError(os.path.join(outputDir, 'masks', id, unannotatedClassName)+' is not a valid path')
              if otherClassNames:
                  if type(otherClassNames) == str:
                      try:
                          os.makedirs(os.path.join(outputDir, 'masks', id, otherClassNames), exist_ok=True)
                      except:
                          raise ValueError(os.path.join(outputDir, 'masks', id, otherClassNames)+' is not a valid path')
                  elif type(otherClassNames) == list:
                      for otherClassName in otherClassNames:
                          if type(otherClassName) != str:
                              raise ValueError('If otherClassNames is a list, all elements of list must be strings')
                          try:
                              os.makedirs(os.path.join(outputDir, 'masks', id, otherClassName), exist_ok=True)
                          except:
                              raise ValueError(os.path.join(outputDir, 'masks', id, otherClassName)+' is not a valid path')
                  else:
                      raise ValueError('otherClassNames must be a string or list of strings')

          if extractSegmentationMasks:
              print("Extracting "+str(len(unannotatedTilesToExtract))+" of "+str(len(unannotatedTileAddresses))+" "+unannotatedClassName+" tiles and segmentation masks...")
          else:
              print("Extracting "+str(len(unannotatedTilesToExtract))+" of "+str(len(unannotatedTileAddresses))+" "+unannotatedClassName+" tiles...")

      # Extract the desired number of unannotated tiles
      channel_sums = np.zeros(3)
      channel_squared_sums = np.zeros(3)
      tileCounter = 0
      normalize_to_1max = transforms.Compose([transforms.ToTensor()])

      #print("Tiles to extract:", unannotatedTilesToExtract)
      for tl in unannotatedTilesToExtract:
          area = self.getTile(tl)
          if (tissueLevelThreshold) and (foregroundLevelThreshold):
              area.write_to_file(os.path.join(outputDir, 'tiles', id, unannotatedClassName,
                  id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
          elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
              area.write_to_file(os.path.join(outputDir, 'tiles', id, unannotatedClassName,
                  id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel.jpg'), Q=100)
          elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
              area.write_to_file(os.path.join(outputDir, 'tiles', id, unannotatedClassName,
                  id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
          else:
              area.write_to_file(os.path.join(outputDir, 'tiles', id, unannotatedClassName,
                  id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize.jpg'), Q=100)

          tileCounter = tileCounter + 1
          if returnTileStats:
              nparea = self.getTile(tl, writeToNumpy=True)[...,:3] # remove transparency channel
              nparea = normalize_to_1max(nparea).numpy() # normalize values from 0-255 to 0-1
              local_channel_sums = np.sum(nparea, axis=(1,2))
              local_channel_squared_sums = np.sum(np.square(nparea), axis=(1,2))

              channel_sums = np.add(channel_sums, local_channel_sums)
              channel_squared_sums = np.add(channel_squared_sums, local_channel_squared_sums)

          if extractSegmentationMasks:
              height = self.tileDictionary[tl]['height']
              mask = Image.new('1', (height, height), 0) # blank mask

              if (tissueLevelThreshold) and (foregroundLevelThreshold):
                  mask.save(os.path.join(outputDir, 'masks', id, unannotatedClassName,
                      id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'))
              elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
                  mask.save(os.path.join(outputDir, 'masks', id, unannotatedClassName,
                      id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_mask.gif'))
              elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
                  mask.save(os.path.join(outputDir, 'masks', id, unannotatedClassName,
                      id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'))
              else:
                  mask.save(os.path.join(outputDir, 'masks', id, unannotatedClassName,
                      id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_mask.jpg'))

      if returnTileStats:
          if slideName:
              return {'slide': slideName,
                      'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                      'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                      'num_tiles': tileCounter}
          else:
              return {'slide': self.slideFileName,
                      'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                      'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                      'num_tiles': tileCounter}
      else:
          return True

  def extractRandomTissueTiles(self, outputDir, slideName=False, numTilesToExtract=100, className='unannotated',
      foregroundLevelThreshold=False, otsuLevelThreshold=False, triangleLevelThreshold=False, tissueLevelThreshold=False, returnTileStats=True, seed=False, extractNonTissue=False):
      """A function to extract randomly selected tiles that don't overlap any
      annotations into directory structure amenable to torch.utils.data.ConcatDataset

      Args:
          outputDir (str): the path to the directory where the tile directory will be stored
          slideName (str, optional): the name of the slide to be used in the file names of the extracted tiles and masks. Default is Slide.slideFileName.
          numTilesToExtract (int, optional): the number of random unannotated tiles to extract. Default is 50.
          className (str, optional): the name that the "class" directory should be called. Default is "unannotated".
          foregroundLevelThreshold (str or int or float, optional): if defined as an int, only extracts tiles with a 0-100 foregroundLevel value less or equal to than the set value (0 is a black tile, 100 is a white tile). Only includes Otsu's method-passing tiles if set to 'otsu', or triangle algorithm-passing tiles if set to 'triangle'. Default is not to filter on foreground at all.
          tissueLevelThreshold (Bool, optional): if defined, only extracts tiles with a 0 to 1 tissueLevel probability greater than or equal to the set value. Default is False.
          returnTileStats (Bool, optional): whether to return the 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation. Default is True.
          seed (int, optional): the random seed to use for reproducible anayses. Default is not to use a seed when randomly selecting tiles.

      Returns:
          dict: A dictionary containing the Slide's name, 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation; if returnTileStats is set to False, True will be returned

      Example:
          channel_data = pathml_slide.extractRandomUnannotatedTiles('/path/to/directory', numTilesToExtract=200, unannotatedClassName="non_metastasis", tissueLevelThreshold=0.995)
      """

      print("Extracting tiles from {}...".format(self.slideFileName))

      if not self.hasTileDictionary():
          raise PermissionError(
              'setTileProperties must be called before extracting tiles')

      if seed:
          if type(seed) != int:
              raise ValueError('Seed must be an integer')
          random.seed(seed)

      # get case ID
      if slideName:
          if type(slideName) != str:
              raise ValueError("slideName must be a string")
          else:
              id = slideName
      else:
          id = self.slideFileId

      if tissueLevelThreshold:
          if ((type(tissueLevelThreshold) != int) and (type(tissueLevelThreshold) != float)) or ((tissueLevelThreshold <= 0) or (tissueLevelThreshold > 1)):
              raise ValueError('tissueLevelThreshold must be a number greater than zero and less than or equal to 1')

      # Collect all tissue tiles
      tissueTileAddresses = self.suitableTileAddresses(tissueLevelThreshold=tissueLevelThreshold, foregroundLevelThreshold=foregroundLevelThreshold, otsuLevelThreshold=otsuLevelThreshold, triangleLevelThreshold=triangleLevelThreshold)

      if type(outputDir)!=list and type(outputDir)!=tuple:
          outputDir = [outputDir]
      if type(numTilesToExtract)!=list and type(numTilesToExtract)!=tuple:
          numTilesToExtract = [numTilesToExtract]
      if (type(numTilesToExtract[0]) != int) or (sum(numTilesToExtract) <= 0):
          raise ValueError('numTilesToExtract must be a integer greater than 0')
      if not len(outputDir)==len(numTilesToExtract):
          raise ValueError('outputDir and numTilesToExtract must have the same number of elements')

      if len(tissueTileAddresses) == 0:
          print('Warning: 0 tissue tiles found; making no tile directories and returning zeroes')
      if len(tissueTileAddresses) < sum(numTilesToExtract):
          print('Warning: '+str(len(tissueTileAddresses))+' tissue tiles found but requested '+str(sum(numTilesToExtract))+' tiles to extract. Extracting all suitable tiles...')
          tissueTilesToExtract = tissueTileAddresses
      else:
          tissueTilesToExtract = random.sample(tissueTileAddresses, sum(numTilesToExtract))

      channel_sums = np.zeros(3)
      channel_squared_sums = np.zeros(3)
      tileCounter = 0
      normalize_to_1max = transforms.Compose([transforms.ToTensor()])

      for dir, num in zip(outputDir, numTilesToExtract):
          # Create empty class tile directories
          if len(tissueTileAddresses) > 0:
              try:
                  os.makedirs(os.path.join(dir, className, id), exist_ok=True)
              except:
                  raise ValueError(os.path.join(dir, className, id)+' is not a valid path')

          # Extract the desired number of unannotated tiles


          #print("Tiles to extract:", tissueTilesToExtract)
          for tl in tissueTilesToExtract[tileCounter:tileCounter+num]:
              area = self.getTile(tl)

              name_str = f"{id}_{className}_{self.tileDictionary[tl]['x']}x_{self.tileDictionary[tl]['y']}y_{self.tileDictionary[tl]['height']}tilesize"
              if tissueLevelThreshold:
                  name_str = f"{name_str}_{round(self.tileDictionary[tl]['tissueLevel']*1000)}tissue"
              if foregroundLevelThreshold:
                  name_str = f"{name_str}_{round(self.tileDictionary[tl]['foregroundLevel']*10)}foreground"
              if otsuLevelThreshold:
                  name_str = f"{name_str}_{round(self.tileDictionary[tl]['otsuLevel']*1000)}otsu"
              if triangleLevelThreshold:
                  name_str = f"{name_str}_{round(self.tileDictionary[tl]['triangleLevel']*1000)}triangle"
              name_str = name_str + ".jpg"

              area.write_to_file(os.path.join(dir, className, id, name_str), Q=100)

              tileCounter = tileCounter + 1
              if returnTileStats:
                  nparea = self.getTile(tl, writeToNumpy=True)[...,:3] # remove transparency channel
                  nparea = normalize_to_1max(nparea).numpy() # normalize values from 0-255 to 0-1
                  local_channel_sums = np.sum(nparea, axis=(1,2))
                  local_channel_squared_sums = np.sum(np.square(nparea), axis=(1,2))

                  channel_sums = np.add(channel_sums, local_channel_sums)
                  channel_squared_sums = np.add(channel_squared_sums, local_channel_squared_sums)

      self.extractedTiles = tissueTilesToExtract

      if returnTileStats:
          if slideName:
              return {'slide': slideName,
                      'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                      'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                      'num_tiles': tileCounter}
          else:
              return {'slide': self.slideFileName,
                      'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                      'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                      'num_tiles': tileCounter}
      else:
          return True
