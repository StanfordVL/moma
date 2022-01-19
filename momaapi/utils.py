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


def is_actor(iid):
  # max number of actors per sub-activity == 52
  return iid.isupper() and (len(iid) == 1 or (len(iid) == 2 and iid[0] == 'A'))


def is_object(iid):
  # max number of objects per sub-activity == 52
  return iid.isnumeric() and 1 <= int(iid) <= 52


def is_entity(iid):
  return is_actor(iid) or is_object(iid)


def are_actors(iids):
  return all([is_actor(iid) for iid in iids])


def are_objects(iids):
  return all([is_object(iid) for iid in iids])


def are_entities(iids):
  return all([is_entity(iid) for iid in iids])


def sort(iids):
  iids_int = [iid for iid in iids if iid.isnumeric()]
  iids_not_int = [iid for iid in iids if not iid.isnumeric()]
  return sorted(iids_not_int, key=lambda x: (len(x), x))+sorted(iids_int, key=int)


def is_consecutive(iids):
  if are_actors(iids):
    iids_consecutive = list(string.ascii_uppercase)
    iids_consecutive = iids_consecutive+['A'+x for x in iids_consecutive]
    return iids == iids_consecutive[:len(iids)]
  elif are_objects(iids):
    iids_consecutive = [str(x) for x in range(1, len(iids)+1)]
    return iids == iids_consecutive
  else:
    return False
