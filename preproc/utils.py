import datetime
import string
import time


def hms2s(hms):
  dt = datetime.datetime.strptime(hms, '%H:%M:%S')
  seconds = datetime.timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second).total_seconds()
  return seconds


def s2hms(s):
  return time.strftime('%H:%M:%S', time.gmtime(s))


def is_hms(hms):
  try:
    datetime.datetime.strptime(hms, '%H:%M:%S')
    return True
  except ValueError:
    return False


def is_actor(id):
  # max number of actors per sub-activity == 52
  # return id.isupper() and (len(id) == 1 or (len(id) == 2 and id[0] == 'A'))
  # max number of actors per sub-activity == 702
  return id.isupper() and (len(id) == 1 or len(id) == 2)


def is_object(id):
  # max number of objects per sub-activity == 52
  # return id.isnumeric() and 1 <= int(id) <= 52
  # max number of objects per sub-activity == 99
  return id.isnumeric() and 1 <= int(id) <= 99


def is_ent(id):
  return is_actor(id) or is_object(id)


def are_actors(ids):
  return all([is_actor(id) for id in ids])


def are_objects(ids):
  return all([is_object(id) for id in ids])


def are_entities(ids):
  return all([is_ent(id) for id in ids])


def sort(ids):
  ids_int = [id for id in ids if id.isnumeric()]
  ids_not_num = [id for id in ids if not id.isnumeric()]
  return sorted(ids_not_num, key=lambda x: (len(x), x))+sorted(ids_int, key=int)


def is_consecutive(ids):
  if are_actors(ids):
    ids_consecutive = list(string.ascii_uppercase)
    ids_consecutive = ids_consecutive+['A'+x for x in ids_consecutive]
    return ids == ids_consecutive[:len(ids)]
  elif are_objects(ids):
    ids_consecutive = [str(x) for x in range(1, len(ids)+1)]
    return ids == ids_consecutive
  else:
    return False
