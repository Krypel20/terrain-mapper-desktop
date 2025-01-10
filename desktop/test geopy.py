from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my_geopy_app")

Latitude = "49.835"
Longitude = "20.5622"

location = geolocator.reverse(Latitude+","+Longitude)
address = location.raw['address']

# Wybierz interesujące Cię elementy
town = address.get('town', '') or address.get('village', '')  # miasto lub wieś

location_str = f"{town}"
print(location_str)