import 'dart:io';

import 'package:flutter/services.dart';

const deviceChannelName = 'nl.wisent.technicalexploration/device';

const deviceChannel = MethodChannel(deviceChannelName);

class DeviceIdentity {
  const DeviceIdentity({
    required this.manufacturer,
    required this.model,
    required this.device,
    required this.product,
    required this.marketName,
  });

  final String manufacturer;
  final String model;
  final String device;
  final String product;
  final String marketName;

  bool get hasMarketName => marketName.trim().isNotEmpty;
}

class BatteryReading {
  const BatteryReading({this.levelPercent, this.tempCelsius});

  final int? levelPercent;
  final double? tempCelsius;

  static const empty = BatteryReading();
}

class ThermalZone {
  const ThermalZone({
    required this.name,
    required this.type,
    required this.raw,
    required this.celsius,
  });

  final String name;
  final String type;
  final int raw;
  final double celsius;
}

class SocThermal {
  const SocThermal({
    required this.legacyPath,
    required this.legacyType,
    required this.legacyRaw,
    required this.legacyCelsius,
    required this.zones,
  });

  final String? legacyPath;
  final String? legacyType;
  final int? legacyRaw;
  final double? legacyCelsius;
  final List<ThermalZone> zones;

  /// The zone the old sysfs reader would have logged as battery temperature.
  String get logLine {
    final legacy = legacyPath == null
        ? 'legacy=none'
        : 'legacy=$legacyPath type=${legacyType ?? ''} raw=$legacyRaw '
            'celsius=${_fixed(legacyCelsius)}';
    final hottest = _hottest(zones);
    final hot = hottest == null
        ? 'hottest=none'
        : 'hottest=${hottest.name} type=${hottest.type} '
            'celsius=${_fixed(hottest.celsius)}';
    return '$legacy | $hot';
  }
}

DeviceIdentity? _cachedIdentity;
bool _loggedMissingMarketName = false;

/// Marketing name plus SKU when the device provides one.
///
/// `samsung` / `Galaxy Z Fold8 Ultra` / `SM-F976B` becomes
/// `Samsung Galaxy Z Fold8 Ultra (SM-F976B)`. With no market name, the SKU
/// is kept (`Samsung SM-F976B`) and the caller logs device and product once.
String formatAndroidDeviceLabel({
  required String manufacturer,
  required String model,
  String? marketName,
}) {
  final brand = _titleCaseBrand(manufacturer.trim());
  final sku = model.trim();
  final market = marketName?.trim() ?? '';
  if (market.isNotEmpty) {
    final hasBrand = brand.isNotEmpty &&
        market.toLowerCase().startsWith(brand.toLowerCase());
    final name = hasBrand || brand.isEmpty ? market : '$brand $market';
    if (sku.isEmpty || name.contains(sku)) return name;
    return '$name ($sku)';
  }
  if (brand.isEmpty) return sku.isEmpty ? 'Android Device' : sku;
  if (sku.isEmpty) return brand;
  if (sku.toLowerCase().contains(brand.toLowerCase())) return sku;
  return '$brand $sku';
}

String _titleCaseBrand(String value) {
  if (value.isEmpty) return value;
  final lower = value.toLowerCase();
  if (value != lower && value != value.toUpperCase()) return value;
  return lower[0].toUpperCase() + lower.substring(1);
}

/// Android `BatteryManager.EXTRA_TEMPERATURE` is tenths of a degree Celsius.
double? celsiusFromBatteryTenths(int? tenths) {
  if (tenths == null) return null;
  return tenths / 10.0;
}

int? batteryPercentFromLevel({required int? level, required int? scale}) {
  if (level == null || scale == null || level < 0 || scale <= 0) return null;
  return ((level * 100) / scale).round();
}

/// Thermal-zone files are millidegrees. Values above 100 are tenths.
double thermalRawToCelsius(num raw) {
  final abs = raw.abs();
  if (abs > 1000) return raw / 1000.0;
  if (abs > 100) return raw / 10.0;
  return raw.toDouble();
}

BatteryReading batteryFromChannelMap(Map<dynamic, dynamic>? map) {
  if (map == null) return BatteryReading.empty;
  final tenths = (map['tempTenths'] as num?)?.toInt();
  final level = (map['level'] as num?)?.toInt();
  final scale = (map['scale'] as num?)?.toInt();
  return BatteryReading(
    levelPercent: batteryPercentFromLevel(level: level, scale: scale),
    tempCelsius: celsiusFromBatteryTenths(tenths),
  );
}

DeviceIdentity identityFromChannelMap(Map<dynamic, dynamic>? map) {
  return DeviceIdentity(
    manufacturer: map?['manufacturer']?.toString() ?? '',
    model: map?['model']?.toString() ?? '',
    device: map?['device']?.toString() ?? '',
    product: map?['product']?.toString() ?? '',
    marketName: map?['marketName']?.toString() ?? '',
  );
}

SocThermal thermalFromChannelMap(Map<dynamic, dynamic>? map) {
  final legacy = map?['legacy'];
  String? path;
  String? type;
  int? raw;
  double? celsius;
  if (legacy is Map) {
    path = legacy['path']?.toString();
    type = legacy['type']?.toString();
    raw = int.tryParse(legacy['raw']?.toString() ?? '');
    if (raw != null) celsius = thermalRawToCelsius(raw);
  }
  final zones = <ThermalZone>[];
  final listed = map?['zones'];
  if (listed is List) {
    for (final item in listed) {
      if (item is! Map) continue;
      final zoneRaw = int.tryParse(item['raw']?.toString() ?? '');
      if (zoneRaw == null) continue;
      zones.add(
        ThermalZone(
          name: item['name']?.toString() ?? '',
          type: item['type']?.toString() ?? '',
          raw: zoneRaw,
          celsius: thermalRawToCelsius(zoneRaw),
        ),
      );
    }
  }
  return SocThermal(
    legacyPath: path,
    legacyType: type,
    legacyRaw: raw,
    legacyCelsius: celsius,
    zones: zones,
  );
}

Future<void> refreshDeviceIdentity() async {
  if (!Platform.isAndroid) return;
  try {
    final map = await deviceChannel.invokeMapMethod<dynamic, dynamic>(
      'getDeviceIdentity',
    );
    _cachedIdentity = identityFromChannelMap(map);
    if (_cachedIdentity != null &&
        !_cachedIdentity!.hasMarketName &&
        !_loggedMissingMarketName) {
      _loggedMissingMarketName = true;
      // ignore: avoid_print
      print(
        '[technical_exploration] device marketname empty; '
        'device=${_cachedIdentity!.device} product=${_cachedIdentity!.product} '
        'model=${_cachedIdentity!.model}',
      );
    }
  } catch (error) {
    // ignore: avoid_print
    print('[technical_exploration] device identity channel failed: $error');
  }
}

String? cachedAndroidDeviceLabel() {
  final identity = _cachedIdentity;
  if (identity == null) return null;
  return formatAndroidDeviceLabel(
    manufacturer: identity.manufacturer,
    model: identity.model,
    marketName: identity.marketName,
  );
}

Future<BatteryReading> readBattery() async {
  if (Platform.isAndroid) {
    try {
      final map = await deviceChannel.invokeMapMethod<dynamic, dynamic>(
        'getBattery',
      );
      return batteryFromChannelMap(map);
    } catch (error) {
      // ignore: avoid_print
      print('[technical_exploration] BatteryManager read failed: $error');
      return BatteryReading.empty;
    }
  }
  if (Platform.isLinux) {
    return BatteryReading(
      levelPercent: _readLinuxBatteryLevel(),
      tempCelsius: _readLinuxBatteryTemp(),
    );
  }
  return BatteryReading.empty;
}

/// SoC zone the previous sysfs reader would have reported, plus the hottest
/// zone. Printed on every run. Not written to the export.
Future<String?> readSocThermalLine() async {
  if (!Platform.isAndroid) return null;
  try {
    final map = await deviceChannel.invokeMapMethod<dynamic, dynamic>(
      'getThermal',
    );
    return thermalFromChannelMap(map).logLine;
  } catch (error) {
    // ignore: avoid_print
    print('[technical_exploration] thermal zone read failed: $error');
    return null;
  }
}

int? _readLinuxBatteryLevel() {
  const paths = [
    '/sys/class/power_supply/battery/capacity',
    '/sys/class/power_supply/BAT0/capacity',
  ];
  for (final path in paths) {
    final val = int.tryParse(_readTrimmed(path) ?? '');
    if (val != null && val >= 0 && val <= 100) return val;
  }
  return null;
}

/// Battery sysfs only. `thermal_zone0` is an SoC sensor and is not a battery
/// temperature.
double? _readLinuxBatteryTemp() {
  const paths = [
    '/sys/class/power_supply/battery/temp',
    '/sys/class/power_supply/BAT0/temp',
  ];
  for (final path in paths) {
    final raw = double.tryParse(_readTrimmed(path) ?? '');
    if (raw == null) continue;
    return thermalRawToCelsius(raw);
  }
  return null;
}

String? _readTrimmed(String path) {
  try {
    final file = File(path);
    if (!file.existsSync()) return null;
    final text = file.readAsStringSync().trim();
    if (text.isEmpty) return null;
    return text;
  } catch (_) {
    return null;
  }
}

ThermalZone? _hottest(List<ThermalZone> zones) {
  ThermalZone? best;
  for (final zone in zones) {
    if (best == null || zone.celsius > best.celsius) best = zone;
  }
  return best;
}

String _fixed(double? value) =>
    value == null ? 'n/a' : value.toStringAsFixed(1);
