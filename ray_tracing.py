def main():
    import numpy as np
    import matplotlib.pyplot as plt

    #Normalization function:
    def normalize(vector):
        return vector / np.linalg.norm(vector)

    #Reflection function:
    def reflected(vector, axis):
        return vector - 2 * np.dot(vector, axis) * axis
    #Ray-sphere intersection:
    def sphere_intersect(center, radius, ray_origin, ray_direction):
        b = 2 * np.dot(ray_direction, ray_origin - center)
        c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None

    #Nearest intersected object(spheres):
    def nearest_intersected_object(spheres, ray_origin, ray_direction):
        distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in spheres]
        nearest_object = None
        min_distance = np.inf
        for index, distance in enumerate(distances):
            if distance and distance < min_distance:
                min_distance = distance
                nearest_object = spheres[index]
        return nearest_object, min_distance

    #Setup:
    width = 1920 #Width pixels
    height = 1080 #Height pixels
    max_depth = 3 #Reflection bounces
    camera = np.array([0, 0, 1]) #Camera position
    light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom
    image = np.zeros((height, width, 3)) #Image array creation

    #Scene creation: Bling-Phong model:
    #Spheres:
    spheres = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100 , 'reflection': 0.5 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100 , 'reflection': 0.5 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100 , 'reflection': 0.5 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.2 }
    ]
    #Triangles:


    #Ray tracing loops:
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):

            #Ray construction:
            pixel = np.array([x, y, 0])#pixel that we are working with
            origin = camera
            direction = normalize(pixel - origin)#direction vector


            color = np.zeros((3))
            reflection = 1
            for k in range(max_depth):
                #Ray-object intersection(for spheres):
                # check for intersections
                nearest_object, min_distance = nearest_intersected_object(spheres, origin, direction)
                if nearest_object is None:
                    break

                # compute intersection point between ray and nearest object
                intersection = origin + min_distance * direction

                #is it lighted the object?:
                normal_to_surface = normalize(intersection - nearest_object['center'])
                shifted_point = intersection + 1e-5 * normal_to_surface
                intersection_to_light = normalize(light['position'] - shifted_point)
                _, min_distance = nearest_intersected_object(spheres, shifted_point, intersection_to_light)
                intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
                is_shadowed = min_distance < intersection_to_light_distance

                if is_shadowed:
                    break

                # Colour application: Blinn-Phong model:
                illumination = np.zeros((3))# RGB
                illumination += nearest_object['ambient'] * light['ambient']# ambient
                illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)# diffuse
                intersection_to_camera = normalize(camera - intersection)# specular
                H = normalize(intersection_to_light + intersection_to_camera)
                illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)

                # reflection
                color += reflection * illumination
                reflection *= nearest_object['reflection']

                # new ray origin and direction
                origin = shifted_point
                direction = reflected(direction, normal_to_surface)

            image[i, j] = np.clip(color, 0, 1)
            print("progress: %d/%d" % (i + 1, height))

    plt.imsave('image.png', image)


if __name__ == "__main__":
    main()
