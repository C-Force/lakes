def load_images(data_dir):
  all_classes, all_images, all_labels = [], [], []
  for i in os.listdir(data_dir):
    current_dir = os.path.join(data_dir, i)
    if os.path.isdir(current_dir):
      all_classes.append(i)
      for img in os.listdir(current_dir):
        if img.endswith('png'):
          all_images.append(os.path.join(current_dir, img))
          all_labels.append(all_classes.index(i))

  temp = np.array([all_images, all_labels]).T
  np.random.shuffle(temp)

  all_images = temp[:, 0]
  all_labels = temp[:, 1].astype(int)

  return all_images, all_labels