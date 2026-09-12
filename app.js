/**
 * Singapore Property Price Predictor - Client-Side Valuation Engine
 * Supports Public HDB Flats and Private Condominiums / ECs.
 * In-Browser Valuation with < 1 ms latency and offline-first resilience.
 */

// Domestic Constants
// Overwritten at load from models/metrics.json; these are only fallbacks for
// file:// mode where fetch() is blocked.
let HDB_MAPE_ERROR_RATE = 0.0722;     // forward-looking MAPE
let CONDO_MAPE_ERROR_RATE = 0.0752;   // forward-looking MAPE

// Feature schema shared with pipeline_public/03_train_model.py (order matters)
const FEATURE_NAMES = [
  'floor_area_sqm', 'storey_avg', 'sale_year', 'sale_month', 'remaining_lease_years',
  'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM',
  'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION',
  'flat_model_2-room', 'flat_model_3Gen', 'flat_model_Adjoined flat', 'flat_model_Apartment',
  'flat_model_DBSS', 'flat_model_Improved', 'flat_model_Improved-Maisonette',
  'flat_model_Model A', 'flat_model_Model A-Maisonette', 'flat_model_Model A2',
  'flat_model_Multi Generation', 'flat_model_New Generation', 'flat_model_Premium Apartment',
  'flat_model_Premium Apartment Loft', 'flat_model_Premium Maisonette', 'flat_model_Simplified',
  'flat_model_Standard', 'flat_model_Terrace', 'flat_model_Type S1', 'flat_model_Type S2',
  'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH',
  'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG',
  'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST',
  'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL',
  'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES',
  'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN',
  // Transport proximity. Must stay last and in this order to match the trainer.
  'mrt_distance_km', 'mrt_walkable'
];

// Pre-packaged scaler parameters (58 features) so engine works 100% offline and on file:// protocol
const DEFAULT_SCALER = {
  scale: [
    0.00297885, 0.02083333, 0.11111111, 0.09090909, 0.01694915, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
  ],
  min: [
    -0.09234436, -0.04166667, -224.11111111, -0.09090909, -0.66101695, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  ]
};

// Singapore Postal Sector (first 2 digits) to Postal District
const SECTOR_TO_DISTRICT = {
  '01': '01', '02': '01', '03': '01', '04': '01', '05': '01', '06': '01',
  '07': '02', '08': '02',
  '14': '03', '15': '03', '16': '03',
  '09': '04', '10': '04',
  '11': '05', '12': '05', '13': '05',
  '17': '06',
  '18': '07', '19': '07',
  '20': '08', '21': '08',
  '22': '09', '23': '09',
  '24': '10', '25': '10', '26': '10', '27': '10',
  '28': '11', '29': '11', '30': '11',
  '31': '12', '32': '12', '33': '12',
  '34': '13', '35': '13', '36': '13', '37': '13',
  '38': '14', '39': '14', '40': '14', '41': '14',
  '42': '15', '43': '15', '44': '15', '45': '15',
  '46': '16', '47': '16', '48': '16',
  '49': '17', '50': '17', '81': '17',
  '51': '18', '52': '18',
  '53': '19', '54': '19', '55': '19', '82': '19',
  '56': '20', '57': '20',
  '58': '21', '59': '21',
  '60': '22', '61': '22', '62': '22', '63': '22', '64': '22',
  '65': '23', '66': '23', '67': '23', '68': '23',
  '69': '24', '70': '24', '71': '24',
  '72': '25', '73': '25',
  '77': '26', '78': '26',
  '75': '27', '76': '27',
  '79': '28', '80': '28'
};

// Fallback mapping from postal sector to HDB town
const SECTOR_TO_HDB_TOWN = {
  '56': 'ANG MO KIO', '57': 'BISHAN', '46': 'BEDOK', '47': 'BEDOK', '48': 'BEDOK',
  '65': 'BUKIT BATOK', '14': 'QUEENSTOWN', '15': 'BUKIT MERAH', '16': 'BUKIT MERAH',
  '67': 'BUKIT PANJANG', '68': 'CHOA CHU KANG', '58': 'BUKIT TIMAH', '59': 'BUKIT TIMAH',
  '01': 'CENTRAL AREA', '02': 'CENTRAL AREA', '03': 'CENTRAL AREA', '04': 'CENTRAL AREA',
  '05': 'CENTRAL AREA', '06': 'CENTRAL AREA', '07': 'CENTRAL AREA', '08': 'CENTRAL AREA',
  '12': 'CLEMENTI', '38': 'GEYLANG', '39': 'GEYLANG', '40': 'GEYLANG',
  '53': 'HOUGANG', '60': 'JURONG EAST', '61': 'JURONG WEST', '64': 'JURONG WEST',
  '31': 'TOA PAYOH', '32': 'KALLANG/WHAMPOA', '33': 'KALLANG/WHAMPOA',
  '44': 'MARINE PARADE', '51': 'PASIR RIS', '82': 'PUNGGOL', '75': 'SEMBAWANG',
  '54': 'SENGKANG', '55': 'SERANGOON', '52': 'TAMPINES', '73': 'WOODLANDS', '76': 'YISHUN'
};

// District Baseline Resale PSF (benchmarked from official URA private residential caveats)
// Fallbacks for file:// protocol where fetch() is blocked
const SECTOR_TO_SEGMENT = {
  "01": "CCR",
  "05": "RCR",
  "06": "CCR",
  "07": "CCR",
  "08": "RCR",
  "09": "RCR",
  "10": "RCR",
  "11": "RCR",
  "12": "OCR",
  "13": "RCR",
  "14": "RCR",
  "15": "RCR",
  "16": "RCR",
  "17": "CCR",
  "18": "RCR",
  "19": "CCR",
  "20": "RCR",
  "21": "RCR",
  "22": "CCR",
  "23": "CCR",
  "24": "CCR",
  "25": "CCR",
  "26": "CCR",
  "27": "CCR",
  "28": "CCR",
  "29": "CCR",
  "30": "CCR",
  "31": "RCR",
  "32": "RCR",
  "33": "RCR",
  "34": "RCR",
  "35": "RCR",
  "36": "RCR",
  "38": "RCR",
  "39": "RCR",
  "40": "RCR",
  "41": "OCR",
  "42": "OCR",
  "43": "RCR",
  "44": "OCR",
  "45": "OCR",
  "46": "OCR",
  "47": "OCR",
  "48": "OCR",
  "50": "OCR",
  "51": "OCR",
  "52": "OCR",
  "53": "OCR",
  "54": "OCR",
  "55": "OCR",
  "56": "OCR",
  "57": "RCR",
  "58": "RCR",
  "59": "RCR",
  "60": "OCR",
  "61": "OCR",
  "63": "OCR",
  "64": "OCR",
  "65": "OCR",
  "66": "OCR",
  "67": "OCR",
  "68": "OCR",
  "73": "OCR",
  "75": "OCR",
  "76": "OCR",
  "77": "OCR",
  "78": "OCR",
  "79": "OCR",
  "80": "OCR",
  "82": "OCR"
};

const DEFAULT_PROJECT_SEGMENTS = {"FRANKEL ESTATE":"OCR","LANDED HOUSING DEVELOPMENT":"OCR","KEMBANGAN VILLAS":"OCR","MAYFAIR RESIDENCES":"OCR","OPERA ESTATE":"OCR","MARINA COLLECTION":"CCR","CLYDESWOOD":"OCR","VAUGHAN GARDEN":"OCR","SUITES @ EASTCOAST":"OCR","THE HILLSIDE":"OCR","VACANZA @ EAST":"OCR","ISLAND COUNTRY VILLAS":"OCR","E MAISON":"OCR","THE GRANDHILL":"RCR","SOUTHAVEN I":"OCR","DOVER PARKVIEW":"RCR","RESIDENTIAL APARTMENTS":"RCR","ONE-NORTH RESIDENCES":"RCR","LAGOON VIEW":"OCR","THE LENOX":"OCR","VILLAS LA VUE":"OCR","VIBES @ EAST COAST":"OCR","LA VIDA @ 130":"OCR","KALEIDO":"OCR","THE PEAK@BALMEG":"RCR","PEOPLE'S PARK COMPLEX":"RCR","SOUTHBANK":"RCR","D' ALMIRA":"OCR","BRADDELL HEIGHTS ESTATE":"OCR","E7 LODGE":"OCR","THE PANORAMA":"OCR","YONG SENG ESTATE":"OCR","SOUTHAVEN II":"OCR","GOLDLEAF GARDENS":"OCR","EAST COAST PARK":"OCR","KIMNAN PARK":"OCR","ST PATRICK'S RESIDENCES":"OCR","SEMBAWANG HILLS ESTATE":"OCR","CHELSEA VILLAGE":"RCR","THE BERTH BY THE COVE":"CCR","PASIR VIEW PARK":"RCR","THE CRISTALLO":"OCR","VILLAGE @ PASIR PANJANG":"RCR","THE ROCHESTER RESIDENCES":"RCR","PARC IMPERIAL":"RCR","EASTBAY":"OCR","SPOTTISWOODE RESIDENCES":"RCR","WHITEHAVEN":"RCR","MANDARIN GARDENS":"OCR","THE RAINTREE":"OCR","WATERLOO VIEW":"RCR","EMERALD GARDEN":"RCR","HORIZON GARDENS":"OCR","SINGA HILLS":"OCR","CLEMENTI PARK":"OCR","LIVINGSTON MANSIONS":"OCR","PEOPLE'S PARK CENTRE":"RCR","THE ARRIS":"RCR","PARC BLEU":"OCR","THE FOLIAGE":"RCR","HILLVIEW GREEN":"OCR","SPRINGDALE CONDOMINIUM":"OCR","PRESTIGE LOFT":"OCR","HERITAGE VIEW":"RCR","SPOTTISWOODE SUITES":"RCR","THE SEAWIND":"OCR","GARDEN PARK RESIDENCES":"OCR","BURLINGTON SQUARE":"RCR","LANDRIDGE CONDOMINIUM":"RCR","CITY GATE":"RCR","FLAMINGO VALLEY":"OCR","THE COAST AT SENTOSA COVE":"CCR","NEPTUNE COURT":"OCR","JERSEY LODGE":"OCR","VENTANA":"RCR","STARVILLE":"OCR","THE MEDLEY":"OCR","SUMMERHILL":"OCR","PARK NATURA":"RCR","BLAIR PLAIN CONSERVATION AREA":"RCR","EUHABITAT":"OCR","KIARA 10":"OCR","THE AMBRA":"OCR","ESCADA VIEW":"OCR","TIERRA VUE":"OCR","SEASCAPE":"CCR","MELINIA COURT":"OCR","EASTERN LAGOON":"OCR","THE OCEANFRONT @ SENTOSA COVE":"CCR","GRANDEUR 8":"OCR","SUNSHINE MANSIONS":"OCR","AXIS @ SIGLAP":"OCR","CASA SARINA":"OCR","DORSETT RESIDENCES":"RCR","SARINA TERRACE":"OCR","ELLIOT AT THE EAST COAST":"OCR","ESPIRA RESIDENCE":"OCR","VILLA DE WEST":"RCR","THE AZURE":"CCR","OCEAN PARK":"OCR","SIGLAP PARK":"OCR","D - HERITAGE APARTMENTS":"OCR","THE MONTAGE":"OCR","HERITAGE EAST":"OCR","ST NICHOLAS VIEW":"OCR","PALM GREEN":"RCR","H RESIDENCES":"OCR","FLORAVIEW":"OCR","THOMSON HILLS ESTATE":"OCR","WELLINGTON PARK":"OCR","SERANGOON PARK":"OCR","ADANA @ THOMSON":"OCR","THE COTZ":"OCR","DUNSFOLD RESIDENCES":"OCR","TROPIKA EAST":"OCR","VIVA VISTA":"RCR","THE VERTE":"OCR","EAST SIGNATURE":"OCR","AQUENE":"OCR","EIGHT @ EAST COAST":"OCR","KEMBANGAN ESTATE":"OCR","LE CONNEY PARK":"OCR","LE MARICAN":"OCR","NUOVO":"OCR","LAVERNE'S LOFT":"OCR","ASTORIA PARK":"OCR","SPOTTISWOODE PARK":"RCR","IDYLLIC RESIDENCES":"OCR","SOMMERVILLE RESIDENCES":"OCR","SUITES@BRADDELL":"OCR","MILFORD VILLAS":"OCR","CALLIDORA VILLE":"OCR","THE HELICONIA":"OCR","PARC PALAIS":"OCR","PALM OASIS":"OCR","SILAHIS VILLE":"OCR","MONT TIMAH":"OCR","KINGSGROVE":"OCR","JI XIANG COURT":"OCR","NATURALIS":"OCR","THE ESTIVA":"RCR","FAR HORIZON GARDENS":"OCR","PALM GALLERIA":"OCR","EASTREES":"OCR","PASIR PANJANG COURT":"RCR","CHATEAU LA SALLE":"OCR","SIGLAP V":"OCR","VILLA MARINA":"OCR","BAYVILLE CONDOMINIUM":"RCR","ST PATRICK'S LOFT":"OCR","THE GLACIER":"OCR","CASA SERENO":"OCR","SENTOSA COVE":"CCR","CENTRO RESIDENCES":"OCR","TURQUOISE":"CCR","TREESCAPE":"OCR","ESPIRA SUITES":"OCR","MABELLE":"OCR","PARC ELEGANCE":"OCR","SUNSHINE PLAZA":"RCR","SYMPHONY HEIGHTS":"OCR","GRAND DUCHESS AT ST PATRICK'S":"OCR","448@EAST COAST":"OCR","SEVENTY SAINT PATRICK'S":"OCR","THE BENCOOLEN":"RCR","CORAL ISLAND":"CCR","RIVERWALK APARTMENTS":"RCR","FORTUNE VIEW":"OCR","SUITES TWENTY-TWO":"OCR","SOMMERVILLE REGENCY":"OCR","BERKELEY RESIDENCES":"OCR","PARK EAST":"OCR","D' PALMA":"OCR","KASARA":"CCR","MATLOCK RESIDENCES":"OCR","HUME PARK I":"OCR","BUONA VISTA GARDENS":"RCR","LIIV RESIDENCES":"RCR","THE NAVIAN":"OCR","ISLAND VIEW":"RCR","SIN CHUAN GARDEN":"OCR","SEASUITES":"RCR","SERENE VIEW MANSIONS":"OCR","THE BEACON":"RCR","SPRINGVALE":"OCR","WHITESCAPE":"OCR","SANDY ISLAND":"CCR","24 ONE RESIDENCES":"RCR","RADIX":"OCR","THE TRUMPS":"OCR","THE MAYLEA":"RCR","SHANGRI-LA PARK":"OCR","PALMERA RESIDENCE":"OCR","MURANO":"RCR","FERNWOOD TERRACES":"OCR","DAISY LODGE":"OCR","ENVIO":"OCR","THE SPECTRUM":"RCR","PROSPERO VILLE":"OCR","FREESIA WOODS":"OCR","KINGSTON TERRACE":"OCR","DE CASALLE":"OCR","UNITED MANSION":"OCR","FERNWOOD TOWERS":"OCR","VERANDA":"OCR","TREASURE PLACE":"RCR","MEDALLION":"OCR","LE WOOD":"OCR","MALAYSIA PARK":"OCR","DAISY PARK":"OCR","RESIDENCE 66":"OCR","MARTIA RESIDENCE":"OCR","CRYSTAL HEIGHTS":"RCR","SOUTH UNION PARK":"OCR","SEVEN PALMS SENTOSA COVE":"CCR","ETON COURT":"OCR","SUITES @ EUNOS":"OCR","HUME PARK II":"OCR","R MAISON":"OCR","ONE CHILTERN":"OCR","LE REVE":"OCR","MERIDIAN 38":"OCR","THE TREELINE":"OCR","BLISS RESIDENCES":"OCR","MENDON SPRING":"RCR","LOTUS VILLE":"OCR","AURALIS":"OCR","BLISS VILLE":"OCR","THE LUCENT":"OCR","COSTA ESTE":"OCR","SUNNY LODGE":"OCR","EAST BAY GARDENS":"OCR","PALM MANSIONS":"RCR","CHWEE CHIAN VIEW":"RCR","SPOTTISWOODE 18":"RCR","THE SCENIC @ BRADDELL":"OCR","OASIS GARDEN":"RCR","TK ONE":"OCR","CRAIG PLACE":"RCR","HORIZON GREEN":"OCR","CRESCENDO PARK":"OCR","LE HILL CONDOMINIUM":"RCR","EAST PALM":"OCR","MERA EAST":"OCR","BIJOU":"RCR","BAROSSA GARDENS":"RCR","THE AMERY":"OCR","DUNSFOLD 18":"OCR","D' CASITA":"OCR","THE NCLAVE":"OCR","GROSVENOR VIEW":"OCR","THE SUNNIDORA":"OCR","THE AMBROSIA":"OCR","THE VILLAS @ SENTOSA COVE":"CCR","OCEAN 8":"CCR","ASTOR":"OCR","LYNWOOD EIGHT":"OCR","COTSWOLD VILLAS":"OCR","PALACIO":"OCR","BANYAN PARK":"RCR","LAGUNA PARK":"OCR","QUBE SUITES":"OCR","TEXTILE CENTRE":"RCR","MAYFLOWER RESIDENCES":"OCR","DAISY SUITES":"OCR","GOLD COAST CONDOMINIUM":"RCR","SHIRO":"OCR","THE LATTIZ":"OCR","HEJI GARDENS":"OCR","HUA XIN COURT":"OCR","THE SUNNIFLORA":"OCR","COASTARINA":"OCR","AQUARINE GARDENS":"OCR","G RESIDENCES":"OCR","CUBIK":"OCR","BRIGHT GARDEN":"OCR","D'SAVILLE":"OCR","D'ECOSIA":"OCR","TERRA VILLAS":"OCR","WINDY HEIGHTS":"OCR","GOLDENHILL PARK CONDOMINIUM":"OCR","SEASIDE PARK":"OCR","THE ACACIAS":"OCR","SUNNY PALMS":"OCR","THE ESPIRA":"OCR","THE RIVERSIDE PIAZZA":"RCR","CHIP THYE GARDEN":"OCR","SIGLAP GARDEN":"OCR","MILL CREEK":"OCR","CATHAY GARDEN":"OCR","IVORY APARTMENTS":"OCR","D'OASIA":"OCR","DENG FU VILLE":"OCR","GOLD LEAF MANSIONS":"OCR","MAYFLOWER GARDENS":"OCR","EASTWOOD GARDENS":"OCR","LYNNSVILLE 331":"RCR","KEMBANGAN SUITES":"OCR","MARTIA 8":"OCR","ST PATRICK'S GREEN":"OCR","GOLDEN HILL ESTATE":"OCR","BANYAN CONDOMINIUM":"RCR","GRACE COURT":"OCR","JOO CHIAT VERBENA":"OCR","SUNSHINE GRANDEUR":"OCR","CHESTER PARK":"RCR","MARINE GARDEN":"OCR","THE SOUND":"OCR","LE MERRITT":"OCR","SILAHIS APARTMENTS":"OCR","PALM VISTA":"OCR","EUNOS PARK":"OCR","PINEHURST CONDOMINIUM":"OCR","WOLSKEL LODGE":"OCR","STILL MANSIONS":"OCR","SUNSET LODGE":"OCR","FRAGRANCE COURT":"RCR","SPRINGDEW TERRACE":"RCR","POH HENG COURT":"OCR","RESIDENCES 88":"OCR","BLU CORAL":"OCR","KEMBANGAN PLAZA":"OCR","LI TONG GARDENS":"OCR","ESPIRA SPRING":"OCR","BREEZEWAYS":"RCR","LEVENUE":"OCR","THE MONDRIAN":"OCR","BELLA VISTA":"OCR","SUNSHINE RESIDENCE":"OCR","TIARA VISTA":"OCR","THE HACIENDA":"OCR","KNOX VIEW":"OCR","FINLAND GARDENS":"OCR","PALISADES":"RCR","GEMINI APARTMENTS":"OCR","POSHGROVE EAST":"OCR","RICCHEZZA":"OCR","SPRINGWOOD":"RCR","EBONY MANSIONS":"OCR","THE EBONY":"OCR","SUNNYVALE RESIDENCES":"OCR","CRESCENDO BUILDING":"OCR","MADEIRA COURT":"OCR","VENTANA VILLAS":"RCR","CAPE ROYALE":"CCR","PALMERA EAST":"OCR","PARADISE ISLAND":"CCR","KINGS APARTMENTS":"OCR","GLAMOUR VILLE":"OCR","ELITE RESIDENCES":"OCR","LA SALLE VILLAS":"OCR","CAMELODGE":"OCR","THE ARECA":"RCR","HAPPY ESTATE":"OCR","TAIPAN GRAND":"OCR","EMPRADO SUITES":"OCR","LUXE VILLE":"RCR","WESTRIDGE":"RCR","SEASIDE RESIDENCES":"OCR","PASIR PANJANG LODGE":"RCR","THE VESTA":"OCR","THE VINES":"OCR","VILLA MARTIA":"OCR","SUNNY PARC":"OCR","THE BALE":"OCR","NATURE MANSIONS":"OCR","THE AZZURO":"OCR","BOWMONT CENTRE":"OCR","HERITAGE RESIDENCES":"OCR","THE PEAK":"RCR","SOMMERVILLE LOFT":"OCR","BOAT QUAY CONSERVATION AREA":"RCR","KEMBANGAN COURT":"OCR","GOODWILL COURT":"OCR","EASTCOVE RESIDENCES":"OCR","WEST SHORE RESIDENCES":"RCR","SRI MENANTI ESTATE":"RCR","MERLOT VILLE":"OCR","WEST SEA GARDENS":"RCR","HORIZON RESIDENCES":"RCR","GRACE GARDEN":"OCR","THE ORIENT":"RCR","ICON @ PASIR PANJANG":"RCR","RESIDENCE 118":"OCR","CANARY VILLE":"OCR","THE BECCAGAYLE":"OCR","UNI BUILDING":"OCR","PEIRCE VIEW":"OCR","THE CASSANDRA":"RCR","STARPOINT":"RCR","D'GALLERY":"OCR","PRESTIGE RESIDENCE":"OCR","PINE COURT":"OCR","TENGGIRIA":"OCR","SARINA PARK":"OCR","CASCATA":"OCR","HARBOUR VIEW GARDENS":"RCR","VICENTA LODGE":"OCR","THE VERANDAH RESIDENCES":"RCR","THE RESIDENCES AT W SINGAPORE SENTOSA COVE":"CCR","MAYFLOWER VILLE":"OCR","CADENCE LIGHT":"OCR","HAPPY PARK":"OCR","DAISY TERRACE":"OCR","ONZE @ TANJONG PAGAR":"RCR","HOMEY GARDENS":"OCR","TG MANSION":"OCR","EAST SHINE":"OCR","CHUVILLE":"RCR","EASTERN RESIDENCE":"OCR","SING LIAN BUILDING":"OCR","77 @ EAST COAST":"OCR","FAME COURT":"OCR","LATTICE ONE":"OCR","PEPYS HILL CONDOMINIUM":"RCR","SAPPHIRE COURT":"OCR","BETA GROVE":"OCR","RESERVOIR VILLAS":"OCR","MANILA GARDENS":"OCR","SIGLAP HILL 33":"OCR","PROSPECT GARDEN":"OCR","WEST PARK REGENCY":"RCR","EALINE PARK":"RCR","ST PATRICK'S ROW":"OCR","EAST GROVE":"OCR","CASTLE LOFT":"OCR","SKY EVERTON":"RCR","URBAN TREASURES":"OCR","SANDERIANA":"OCR","SENG YONG VILLAS":"OCR","RESIDENCE TWENTY-TWO":"OCR","SUNSHINE LOFT":"OCR","FIVENINE":"OCR","SIGLAP LODGE":"OCR","BEAU VISTA":"OCR","BUONA LODGE":"RCR","SERENITY BREEZE":"OCR","TEDGE":"OCR","E-SPACE":"OCR","THE DOMAIN":"OCR","WEST-N":"RCR","ONE-NORTH EDEN":"RCR","KENTVIEW PARK":"RCR","KENT RIDGE HILL RESIDENCES":"RCR","CAYMAN RESIDENCES":"OCR","SENG HUA GARDENS":"OCR","CANDEVILLE":"OCR","ST PATRICK'S COURT":"OCR","RYMDEN 77":"OCR","THE GRANDIFLORA":"OCR","SENYUM RESIDENCES":"OCR","FOOK HAI BUILDING":"RCR","THE CHAMPAGNE":"OCR","TELOK MANSION":"OCR","BAYWIND RESIDENCES":"OCR","NORMANTON PARK":"RCR","PARC REGENCY":"RCR","CHARIS LODGE":"RCR","WESTPOINT CONDOMINIUM":"RCR","GLORIA MANSION":"RCR","ROYAL PALM MANSIONS":"RCR","THE ESTRELLA":"RCR","JUBILEE RESIDENCE":"RCR","D' HILLSIDE LOFT":"RCR","PLEASANT HILL LODGE":"RCR","SARHAD VILLE":"RCR","RIA APARTMENTS":"RCR"};

const DEFAULT_CONDO_LOOKUP = {
  "569788": {
    "postal_code": "569788",
    "project_name": "NUOVO",
    "property_type": "EC",
    "dwelling_units": 101,
    "x": 28976.329687,
    "y": 40805.664052,
    "longitude": 103.842091644608,
    "latitude": 1.38530643029336
  },
  "569787": {
    "postal_code": "569787",
    "project_name": "NUOVO",
    "property_type": "EC",
    "dwelling_units": 100,
    "x": 28983.0899504,
    "y": 40751.451369,
    "longitude": 103.842152389239,
    "latitude": 1.38481614983743
  },
  "569786": {
    "postal_code": "569786",
    "project_name": "NUOVO",
    "property_type": "EC",
    "dwelling_units": 96,
    "x": 29011.7938058,
    "y": 40701.8409935,
    "longitude": 103.84241031298,
    "latitude": 1.38436749017877
  },
  "787056": {
    "postal_code": "787056",
    "project_name": "CASTLE GREEN",
    "property_type": "Non-Landed",
    "dwelling_units": 168,
    "x": 28883.792016,
    "y": 40949.9035556,
    "longitude": 103.841260127243,
    "latitude": 1.3866108835755
  },
  "787057": {
    "postal_code": "787057",
    "project_name": "CASTLE GREEN",
    "property_type": "Non-Landed",
    "dwelling_units": 168,
    "x": 28918.2664625,
    "y": 40864.371137,
    "longitude": 103.841569904374,
    "latitude": 1.38583735724226
  },
  "787058": {
    "postal_code": "787058",
    "project_name": "CASTLE GREEN",
    "property_type": "Non-Landed",
    "dwelling_units": 164,
    "x": 28823.9892602,
    "y": 40819.0590574,
    "longitude": 103.84072275048,
    "latitude": 1.38542757456627
  },
  "538804": {
    "postal_code": "538804",
    "project_name": "THE FLORIDA",
    "property_type": "EC",
    "dwelling_units": 123,
    "x": 35460.165933,
    "y": 39465.3074436,
    "longitude": 103.900353496505,
    "latitude": 1.37318379482838
  },
  "018987": {
    "postal_code": "018987",
    "project_name": "THE SAIL @ MARINA BAY",
    "property_type": "Non-Landed",
    "dwelling_units": 681,
    "x": 30152.3828113,
    "y": 29246.5049786,
    "longitude": 103.85265856205,
    "latitude": 1.28076943496046
  }
};

const DISTRICT_BASE_PSF = {
  '01': 2650, '02': 2550, '03': 2100, '04': 2350, '05': 1850,
  '06': 2700, '07': 2400, '08': 1880, '09': 2850, '10': 2750,
  '11': 2600, '12': 1800, '13': 1850, '14': 1750, '15': 2150,
  '16': 1650, '17': 1420, '18': 1480, '19': 1580, '20': 1950,
  '21': 2050, '22': 1450, '23': 1520, '24': 1350, '25': 1380,
  '26': 1500, '27': 1400, '28': 1480
};

// HDB Town Baseline Resale PSF (benchmarked from 240k HDB resale transactions)
const HDB_TOWN_BASE_PSF = {
  'ANG MO KIO': 712, 'BEDOK': 687, 'BISHAN': 769, 'BUKIT BATOK': 584,
  'BUKIT MERAH': 883, 'BUKIT PANJANG': 566, 'BUKIT TIMAH': 723, 'CENTRAL AREA': 698,
  'CHOA CHU KANG': 538, 'CLEMENTI': 797, 'GEYLANG': 765, 'HOUGANG': 607,
  'JURONG EAST': 542, 'JURONG WEST': 542, 'KALLANG/WHAMPOA': 777, 'MARINE PARADE': 702,
  'PASIR RIS': 626, 'PUNGGOL': 580, 'QUEENSTOWN': 815, 'SEMBAWANG': 547,
  'SENGKANG': 588, 'SERANGOON': 664, 'TAMPINES': 635, 'TOA PAYOH': 767,
  'WOODLANDS': 551, 'YISHUN': 555
};

// HDB Flat Type & Model multipliers
const HDB_FLAT_TYPE_FACTOR = {
  '1 ROOM': 1.85, '2 ROOM': 1.25, '3 ROOM': 1.05, '4 ROOM': 1.00,
  '5 ROOM': 1.01, 'EXECUTIVE': 1.00, 'MULTI-GENERATION': 1.02
};

const HDB_FLAT_MODEL_FACTOR = {
  'DBSS': 1.22, 'Terrace': 1.35, 'Maisonette': 1.12, 'Model A-Maisonette': 1.12,
  'Improved-Maisonette': 1.12, 'Premium Maisonette': 1.15, 'Premium Apartment': 1.04,
  'Premium Apartment Loft': 1.15, 'Type S1': 1.10, 'Type S2': 1.12,
  '3Gen': 1.02, 'Adjoined flat': 1.05, 'Apartment': 1.00, 'Model A': 1.00,
  'Model A2': 0.98, 'Improved': 1.00, 'New Generation': 0.98, 'Simplified': 0.94,
  'Standard': 0.92, '2-room': 0.96, 'Multi Generation': 1.02
};

// Published accuracy figures, regenerated by the trainers on every run and
// mirrored here so the About panel still renders when the page is opened
// straight from disk, where fetch() is blocked. models/metrics.json wins
// whenever it can be loaded.
const DEFAULT_METRICS = {
  "condo": {
    "annual_growth": 0.073128,
    "features": 45,
    "forecast_base_year": 2025,
    "generated_at": "2026-09-12",
    "holdout_samples": 5458,
    "index_source": "URA Property Price Index of Non-landed Properties by Locality (CCR/RCR/OCR), quarterly",
    "label": "Private Non-Landed Model (Condo, Apartment, EC)",
    "latest_year": 2026,
    "mae": 136553.2,
    "mape": 6.27,
    "projects": 2204,
    "r2": 0.9597,
    "records": 71538,
    "rmse": 269177.82,
    "segment_growth": {
      "CCR": 0.035272,
      "OCR": 0.074701,
      "RCR": 0.058042
    },
    "source": "Urban Redevelopment Authority (URA Caveats)",
    "trees": 77,
    "validation": "chronological (validate on 2026+)"
  },
  "generated_at": "2026-09-12",
  "hdb": {
    "annual_growth": 0.067346,
    "features": 60,
    "forecast_base_year": 2025,
    "generated_at": "2026-09-12",
    "holdout_samples": 18278,
    "index_source": "HDB Resale Price Index (1Q2009=100), quarterly",
    "label": "Public Housing (HDB Flat Resale Model)",
    "latest_year": 2026,
    "mae": 31089.56,
    "mape": 4.63,
    "r2": 0.9553,
    "records": 240345,
    "rmse": 45074.6,
    "source": "Housing & Development Board (Data.gov.sg)",
    "trees": 599,
    "validation": "chronological (validate on 2026+)"
  }
};

// Global State
let currentPropertyClass = 'public'; // 'public' | 'private'
let modelData = null;                // XGBoost 600-tree ensemble for HDB
let scalerData = DEFAULT_SCALER;     // Default fallback scaler for HDB
let condoModelData = null;           // XGBoost 285-tree ensemble for Private Condos
let condoScalerData = null;          // Private condo scaler
let postalLookup = {};               // HDB block dictionary
let condoLookup = (typeof DEFAULT_CONDO_LOOKUP !== 'undefined') ? DEFAULT_CONDO_LOOKUP : {};
let currentUnit = 'sqm';             // 'sqm' | 'sqft'
let isXgbReady = false;              // True once the HDB tree ensemble JSON has loaded
let isCondoXgbReady = false;         // True once Condo trees JSON loaded
let condoProjectEncoding = null;     // {project -> mean PSF} target encodings
let publishedMetrics = null;         // models/metrics.json, rendered into the About panel
let mrtProximity = null;             // postal / 'BLK|TOWN' -> km to nearest MRT exit
let currentCondoStreet = '';         // street of the resolved condo, for encoding

// Cached Element References
let el = {};

function initElements() {
  el = {
    // Class Switcher
    btnClassPublic: document.getElementById('btnClassPublic'),
    btnClassPrivate: document.getElementById('btnClassPrivate'),
    sectionPublic: document.getElementById('section-public'),
    sectionPrivate: document.getElementById('section-private'),

    // Public HDB Inputs
    postalInput: document.getElementById('postalInput'),
    postalBadge: document.getElementById('postalBadge'),
    townSelect: document.getElementById('townSelect'),
    flatTypeSelect: document.getElementById('flatTypeSelect'),
    flatModelSelect: document.getElementById('flatModelSelect'),
    leaseYearInput: document.getElementById('leaseYearInput'),

    // Private Condo Inputs
    condoPostalInput: document.getElementById('condoPostalInput'),
    condoPostalBadge: document.getElementById('condoPostalBadge'),
    condoProjectInput: document.getElementById('condoProjectInput'),
    condoTypeSelect: document.getElementById('condoTypeSelect'),
    condoDistrictSelect: document.getElementById('condoDistrictSelect'),
    condoSegmentBadge: document.getElementById('condoSegmentBadge'),
    condoTenureSelect: document.getElementById('condoTenureSelect'),
    condoLeaseGroup: document.getElementById('condoLeaseGroup'),
    condoLeaseYearInput: document.getElementById('condoLeaseYearInput'),

    // Common Inputs
    floorAreaSlider: document.getElementById('floorAreaSlider'),
    floorAreaVal: document.getElementById('floorAreaVal'),
    floorAreaBounds: document.getElementById('floorAreaBounds'),
    storeySlider: document.getElementById('storeySlider'),
    storeyVal: document.getElementById('storeyVal'),
    saleYearSelect: document.getElementById('saleYearSelect'),
    saleMonthSelect: document.getElementById('saleMonthSelect'),

    // Outputs
    priceHeroLabel: document.getElementById('priceHeroLabel'),
    propertySummaryBadge: document.getElementById('propertySummaryBadge'),
    priceDisplay: document.getElementById('priceDisplay'),
    priceRangeDisplay: document.getElementById('priceRangeDisplay'),
    metricPsm: document.getElementById('metricPsm'),
    metricPsf: document.getElementById('metricPsf'),
    metricLeaseRem: document.getElementById('metricLeaseRem'),
    metricLatency: document.getElementById('metricLatency'),
    engineStatus: document.getElementById('engineStatus'),

    // Controls
    themeToggleBtn: document.getElementById('themeToggleBtn'),
    sqmBtn: document.getElementById('sqmBtn'),
    sqftBtn: document.getElementById('sqftBtn')
  };
}

// Calibrated Baseline Valuation for HDB (runs in 0 ms, works 100% offline)
function evaluateHdbBaseline(floorAreaSqm, storey, saleYear, remainingLease, town, flatType, flatModel) {
  const startTime = performance.now();
  const areaSqft = floorAreaSqm * 10.7639;
  const basePsf = HDB_TOWN_BASE_PSF[town] || 640;
  const typeFactor = HDB_FLAT_TYPE_FACTOR[flatType] || 1.0;
  const modelFactor = HDB_FLAT_MODEL_FACTOR[flatModel] || 1.0;

  // Storey factor
  let storeyFactor = 1.0;
  if (storey <= 3) storeyFactor = 0.97;
  else if (storey <= 10) storeyFactor = 0.98 + (storey - 4) * 0.003;
  else if (storey <= 25) storeyFactor = 1.00 + (storey - 10) * 0.004;
  else storeyFactor = 1.06 + (storey - 25) * 0.003;

  // Remaining lease factor relative to 70-year benchmark
  const leaseFactor = Math.pow(Math.min(remainingLease, 99.0) / 70.0, 0.42);

  // Time factor (2.5% annual baseline appreciation)
  const timeFactor = Math.pow(1.025, saleYear - 2026);

  const estimatedPsf = basePsf * typeFactor * modelFactor * storeyFactor * leaseFactor * timeFactor;
  const totalPrice = estimatedPsf * areaSqft;
  const latency = (performance.now() - startTime).toFixed(2);

  return {
    prediction: totalPrice,
    latencyMs: latency < 0.1 ? '< 1' : latency,
    isEnsemble: false
  };
}

// In-Browser XGBoost Evaluation for Public HDB (600 decision trees)
function evaluateHdbModel(inputFeatures) {
  if (!modelData || !scalerData) return null;

  const startTime = performance.now();

  // Clamp the sale year to the model's training range and recover the growth
  // factor to apply afterwards (index 2 = sale_year, index 3 = sale_month).
  const adj = getForecastAdjustment(scalerData, inputFeatures[2], inputFeatures[3]);
  const modelInputs = Float32Array.from(inputFeatures);
  modelInputs[2] = adj.modelYear;
  modelInputs[3] = adj.modelMonth;

  // 1. MinMax Scale Inputs: x * scale + min
  const featureCount = FEATURE_NAMES.length;
  const scaled = new Float32Array(featureCount);
  for (let i = 0; i < featureCount; i++) {
    scaled[i] = modelInputs[i] * scalerData.scale[i] + scalerData.min[i];
  }

  // 2. Traverse Decision Trees
  let totalScore = modelData.b;
  const trees = modelData.t;
  const numTrees = trees.length;

  for (let t = 0; t < numTrees; t++) {
    const tree = trees[t];
    let node = 0;
    while (tree.l[node] !== -1) {
      const featIdx = tree.f[node];
      if (scaled[featIdx] < tree.c[node]) {
        node = tree.l[node];
      } else {
        node = tree.r[node];
      }
    }
    totalScore += tree.w[node];
  }

  const latency = (performance.now() - startTime).toFixed(2);
  return {
    prediction: totalScore * adj.growthFactor,
    latencyMs: latency,
    isEnsemble: true,
    yearsProjected: adj.yearsProjected,
    growthRate: adj.growthRate
  };
}

// In-Browser XGBoost Evaluation for Private Condominiums (285 decision trees)
function evaluateCondoTreeModel() {
  if (!condoModelData || !condoScalerData) return null;

  const startTime = performance.now();
  const floorAreaSqm = parseFloat(el.floorAreaSlider.value) || 90;
  const floorAreaSqft = floorAreaSqm * 10.7639;
  const storey = parseFloat(el.storeySlider.value) || 8;
  const district = el.condoDistrictSelect ? el.condoDistrictSelect.value : '19';
  const propType = (el.condoTypeSelect ? el.condoTypeSelect.value : 'CONDOMINIUM').toUpperCase();
  const tenure = el.condoTenureSelect ? el.condoTenureSelect.value : '99';
  const saleYear = parseFloat(el.saleYearSelect.value) || 2026;
  const saleMonth = parseFloat(el.saleMonthSelect ? el.saleMonthSelect.value : 6) || 6;

  let isFreehold = 0, is99Yr = 0, is999Yr = 0, remainingLease = 99.0;
  if (tenure === 'FREEHOLD') {
    isFreehold = 1;
    remainingLease = 999.0;
  } else if (tenure === '999') {
    is999Yr = 1;
    remainingLease = 999.0;
  } else {
    is99Yr = 1;
    const commenceYear = parseFloat(el.condoLeaseYearInput.value) || 2000;
    remainingLease = Math.max(1.0, Math.min(99.0, (commenceYear + 99.0) - saleYear));
  }

  // Project identity is the strongest single price signal available (a named
  // development explains ~87% of PSF variance versus ~47% for its district), so
  // it is fed in as a smoothed mean-price encoding rather than 2,200 dummies.
  // An unrecognised or blank project falls back to the global mean, which
  // reduces the model to roughly its district-level behaviour.
  const projectName = el.condoProjectInput
    ? el.condoProjectInput.value.trim().toUpperCase() : '';

  // Districts with no transactions fall back to the nearest same-segment one.
  const resolvedDistrict = resolveModelDistrict(district);
  const segment = getMarketSegment(resolvedDistrict.code, projectName);

  // Clamp sale_year into the training range; growth is applied to the output.
  const adj = getForecastAdjustment(condoScalerData, saleYear, saleMonth, segment);
  const enc = condoProjectEncoding;
  // Unknown project: fall back to this district's mean rather than the island
  // mean, which is pulled upward by central-region sales.
  const districtFallback = (enc && enc.project_psf_by_district)
    ? enc.project_psf_by_district[resolvedDistrict.code]
    : undefined;
  const projectPsf = (enc && enc.project_psf && enc.project_psf[projectName] !== undefined)
    ? enc.project_psf[projectName]
    : (districtFallback !== undefined
        ? districtFallback
        : (enc ? enc.project_psf_global : 0));
  const projectTxns = (enc && enc.project_txn_count && enc.project_txn_count[projectName] !== undefined)
    ? enc.project_txn_count[projectName] : 0;
  const streetPsf = (enc && enc.street_psf && enc.street_psf[currentCondoStreet] !== undefined)
    ? enc.street_psf[currentCondoStreet]
    : (enc ? enc.street_psf_global : 0);
  const projectMatched = !!(enc && enc.project_psf && enc.project_psf[projectName] !== undefined);

  const featMap = {
    'area': floorAreaSqm,
    'floor_mid': storey,
    'remaining_lease': remainingLease,
    'sale_year': adj.modelYear,
    'sale_month': adj.modelMonth,
    'project_psf': projectPsf,
    'street_psf': streetPsf,
    'project_txn_count': projectTxns,
    'area_x_floor': floorAreaSqm * storey,
    'is_freehold': isFreehold,
    'is_99_yr': is99Yr,
    'is_999_yr': is999Yr,
    'prop_APARTMENT': (propType === 'APARTMENT') ? 1.0 : 0.0,
    'prop_CONDOMINIUM': (propType === 'CONDOMINIUM') ? 1.0 : 0.0,
    'prop_EXECUTIVE CONDOMINIUM': (propType.includes('EXECUTIVE') || propType === 'EC') ? 1.0 : 0.0,
    'segment_CCR': (segment === 'CCR') ? 1.0 : 0.0,
    'segment_OCR': (segment === 'OCR') ? 1.0 : 0.0,
    'segment_RCR': (segment === 'RCR') ? 1.0 : 0.0,
    [`dist_${resolvedDistrict.code}`]: 1.0
  };

  const numFeatures = condoScalerData.features.length;
  const scaled = new Float32Array(numFeatures);
  for (let i = 0; i < numFeatures; i++) {
    const fName = condoScalerData.features[i];
    const rawVal = Object.prototype.hasOwnProperty.call(featMap, fName) ? featMap[fName] : 0.0;
    scaled[i] = rawVal * condoScalerData.scale[i] + condoScalerData.min[i];
  }

  let totalScore = condoModelData.b;
  const trees = condoModelData.t;
  const numTrees = trees.length;

  for (let t = 0; t < numTrees; t++) {
    const tree = trees[t];
    let node = 0;
    while (tree.l[node] !== -1) {
      const featIdx = tree.f[node];
      if (scaled[featIdx] < tree.c[node]) {
        node = tree.l[node];
      } else {
        node = tree.r[node];
      }
    }
    totalScore += tree.w[node];
  }

  const projectedPrice = totalScore * adj.growthFactor;
  const latency = (performance.now() - startTime).toFixed(2);
  const estimatedPsf = projectedPrice / floorAreaSqft;

  return {
    prediction: projectedPrice,
    latencyMs: latency < 0.1 ? '< 1' : latency,
    psf: estimatedPsf,
    psm: projectedPrice / floorAreaSqm,
    remainingLease,
    tenure,
    isEnsemble: true,
    yearsProjected: adj.yearsProjected,
    growthRate: adj.growthRate,
    districtSubstituted: resolvedDistrict.substituted ? resolvedDistrict.code : null,
    projectMatched,
    projectName
  };
}

// Standalone Private Condo Valuation Engine (hedonic pricing baseline fallback)
function evaluateCondoModel() {
  const startTime = performance.now();

  const floorAreaSqm = parseFloat(el.floorAreaSlider.value) || 90;
  const floorAreaSqft = floorAreaSqm * 10.7639;
  const storey = parseFloat(el.storeySlider.value) || 8;
  const district = el.condoDistrictSelect ? el.condoDistrictSelect.value : '19';
  const propType = el.condoTypeSelect ? el.condoTypeSelect.value : 'CONDOMINIUM';
  const tenure = el.condoTenureSelect ? el.condoTenureSelect.value : '99';
  const saleYear = parseFloat(el.saleYearSelect.value) || 2026;

  // 1. District Base PSF
  const basePsf = DISTRICT_BASE_PSF[district] || 1650;

  // 2. Property Type Factor
  let propFactor = 1.0;
  if (propType === 'APARTMENT') propFactor = 0.94;
  else if (propType === 'EXECUTIVE CONDOMINIUM' || propType === 'EC') propFactor = 0.83;

  // 3. Tenure Factor
  let tenureFactor = 1.0;
  let remainingLease = 99.0;
  if (tenure === 'FREEHOLD') {
    tenureFactor = 1.15;
    remainingLease = 999.0;
  } else if (tenure === '999') {
    tenureFactor = 1.12;
    remainingLease = 999.0;
  } else {
    // 99-year leasehold
    const commenceYear = parseFloat(el.condoLeaseYearInput.value) || 2000;
    remainingLease = Math.max(10.0, Math.min(99.0, (commenceYear + 99.0) - saleYear));
    tenureFactor = Math.sqrt(remainingLease / 99.0);
  }

  // 4. Storey Premium
  let storeyFactor = 1.0;
  if (storey <= 5) {
    storeyFactor = 0.97;
  } else if (storey <= 15) {
    storeyFactor = 1.0 + (storey - 6) * 0.005;
  } else if (storey <= 30) {
    storeyFactor = 1.05 + (storey - 16) * 0.006;
  } else {
    storeyFactor = 1.14 + (storey - 31) * 0.004;
  }

  // 5. Projected Time Factor (2.5% annual baseline appreciation from 2026)
  const timeFactor = Math.pow(1.025, saleYear - 2026);

  // Final Price Calculation
  const estimatedPsf = basePsf * propFactor * tenureFactor * storeyFactor * timeFactor;
  const totalPrice = estimatedPsf * floorAreaSqft;
  const latency = (performance.now() - startTime).toFixed(2);

  return {
    prediction: totalPrice,
    latencyMs: latency < 0.1 ? '< 1' : latency,
    psf: estimatedPsf,
    psm: totalPrice / floorAreaSqm,
    remainingLease,
    tenure
  };
}

// Build the HDB feature vector (length driven by FEATURE_NAMES)
function collectHdbFeatures() {
  const features = new Float32Array(FEATURE_NAMES.length);

  const floorAreaSqm = parseFloat(el.floorAreaSlider.value) || 90;
  const storeyAvg = parseFloat(el.storeySlider.value) || 8;
  const saleYear = parseFloat(el.saleYearSelect.value) || 2026;
  const saleMonth = parseFloat(el.saleMonthSelect.value) || 9;
  const leaseCommenceYear = parseFloat(el.leaseYearInput.value) || 1995;

  const remainingLeaseYears = Math.min(Math.max((leaseCommenceYear + 99.0) - saleYear, 10.0), 99.0);

  features[0] = floorAreaSqm;
  features[1] = storeyAvg;
  features[2] = saleYear;
  features[3] = saleMonth;
  features[4] = remainingLeaseYears;

  const selectedFlatType = el.flatTypeSelect.value;
  const selectedFlatModel = el.flatModelSelect.value;
  const selectedTown = el.townSelect.value;

  const targetFlatTypeCol = `flat_type_${selectedFlatType}`;
  const targetFlatModelCol = `flat_model_${selectedFlatModel}`;
  const targetTownCol = `town_${selectedTown}`;

  for (let i = 5; i < FEATURE_NAMES.length; i++) {
    const featName = FEATURE_NAMES[i];
    if (featName === targetFlatTypeCol || featName === targetFlatModelCol || featName === targetTownCol) {
      features[i] = 1.0;
    }
  }

  // Transport proximity, looked up from the block the user entered. Without a
  // resolved block we fall back to the island median so the flat is priced as
  // typical rather than as unusually near or far from a station.
  const mrtKm = lookupMrtDistance();
  const mrtIdx = FEATURE_NAMES.indexOf('mrt_distance_km');
  const walkIdx = FEATURE_NAMES.indexOf('mrt_walkable');
  if (mrtIdx !== -1) features[mrtIdx] = mrtKm;
  if (walkIdx !== -1) features[walkIdx] = mrtKm <= MRT_WALKABLE_KM ? 1.0 : 0.0;

  return { vector: features, floorAreaSqm, remainingLeaseYears, storeyAvg, saleYear,
           selectedTown, selectedFlatType, selectedFlatModel, mrtKm };
}

// A forecast past the training cutoff rests on an assumed growth rate. Say so
// next to the number rather than presenting a projection as a measurement.
function forecastSuffix(res) {
  if (!res || !res.yearsProjected || res.yearsProjected <= 0) return '';
  const rate = (res.growthRate * 100).toFixed(1);
  const yrs = res.yearsProjected < 1
    ? `${Math.round(res.yearsProjected * 12)} mth`
    : `${res.yearsProjected.toFixed(1)} yr`;
  return ` • index-adjusted forward valuation: +${yrs} @ ${rate}%/yr`;
}

// The +/- band is the model's measured holdout error, which describes a
// comparable valuation at the training market level. Once an estimate is
// projected forward it also carries the index assumption, which is not a
// measured quantity - so the band is labelled as covering only the valuation
// part rather than the forecast as a whole.
function confidenceLabel(res, ratePct) {
  const band = `±${ratePct.toFixed(1)}%`;
  return (res && res.yearsProjected > 0) ? `${band} valuation error` : band;
}

// Master Prediction Runner - Executes immediately without blocking
function runPrediction() {
  if (!el.floorAreaSlider) return;

  const floorAreaSqm = parseFloat(el.floorAreaSlider.value) || 90;
  const floorAreaSqft = floorAreaSqm * 10.7639;

  if (currentPropertyClass === 'public') {
    // ----------------------------------------------------
    // Public HDB Flat Valuation
    // ----------------------------------------------------
    const { vector, remainingLeaseYears, storeyAvg, saleYear, selectedTown, selectedFlatType, selectedFlatModel } = collectHdbFeatures();

    el.priceHeroLabel.textContent = 'Estimated HDB Resale Value';
    el.propertySummaryBadge.textContent = `🏢 ${selectedFlatType} • ${selectedFlatModel} • ${selectedTown}`;

    let res = null;
    if (isXgbReady && modelData) {
      res = evaluateHdbModel(vector);
    }

    // Graceful baseline fallback if XGBoost tree file is loading or offline
    if (!res) {
      res = evaluateHdbBaseline(floorAreaSqm, storeyAvg, saleYear, remainingLeaseYears, selectedTown, selectedFlatType, selectedFlatModel);
    }

    const price = Math.max(150000, res.prediction);
    const lowPrice = price * (1.0 - HDB_MAPE_ERROR_RATE);
    const highPrice = price * (1.0 + HDB_MAPE_ERROR_RATE);
    const band = confidenceLabel(res, HDB_MAPE_ERROR_RATE * 100);
    const modelTag = res.isEnsemble ? `${band} • XGBoost Ensemble` : `${band} • Statistical ML`;

    el.priceDisplay.textContent = `S$ ${formatNumber(Math.round(price))}`;
    el.priceRangeDisplay.textContent = `Est. Range: S$ ${formatNumber(Math.round(lowPrice))} – S$ ${formatNumber(Math.round(highPrice))} (${modelTag})${forecastSuffix(res)}`;

    const psm = price / floorAreaSqm;
    const psf = price / floorAreaSqft;

    el.metricPsm.textContent = `S$ ${formatNumber(Math.round(psm))}`;
    el.metricPsf.textContent = `S$ ${formatNumber(Math.round(psf))}`;
    el.metricLeaseRem.textContent = `${remainingLeaseYears.toFixed(1)} yrs`;
    el.metricLatency.textContent = `${res.latencyMs} ms`;
  } else {
    // ----------------------------------------------------
    // Private Condominium Valuation
    // ----------------------------------------------------
    const projName = (el.condoProjectInput && el.condoProjectInput.value.trim()) || 'Private Non-Landed';
    const districtCode = el.condoDistrictSelect ? el.condoDistrictSelect.value : '19';
    const segment = getMarketSegment(districtCode, projName);
    const tenureVal = el.condoTenureSelect ? el.condoTenureSelect.value : '99';
    const tenureLabel = tenureVal === 'FREEHOLD' ? 'Freehold' : (tenureVal === '999' ? '999-Yr' : '99-Yr');

    el.priceHeroLabel.textContent = 'Estimated Private Condo Valuation';
    el.propertySummaryBadge.textContent = `🏙️ ${projName} • D${districtCode} (${segment}) • ${tenureLabel}`;

    const treeRes = evaluateCondoTreeModel();
    const res = treeRes || evaluateCondoModel();
    const price = Math.max(300000, res.prediction);
    const lowPrice = price * (1.0 - CONDO_MAPE_ERROR_RATE);
    const highPrice = price * (1.0 + CONDO_MAPE_ERROR_RATE);

    const modelBadge = res.isEnsemble ? 'URA XGBoost ML' : 'URA Hedonic Model';
    el.priceDisplay.textContent = `S$ ${formatNumber(Math.round(price))}`;
    const condoBand = confidenceLabel(res, CONDO_MAPE_ERROR_RATE * 100);
    el.priceRangeDisplay.textContent = `Est. Range: S$ ${formatNumber(Math.round(lowPrice))} – S$ ${formatNumber(Math.round(highPrice))} (${condoBand} • ${modelBadge})${forecastSuffix(res)}`;

    el.metricPsm.textContent = `S$ ${formatNumber(Math.round(res.psm))}`;
    el.metricPsf.textContent = `S$ ${formatNumber(Math.round(res.psf))}`;
    el.metricLeaseRem.textContent = res.tenure === 'FREEHOLD' ? 'Freehold' : (res.tenure === '999' ? '999 yrs' : `${res.remainingLease.toFixed(1)} yrs`);
    el.metricLatency.textContent = `${res.latencyMs} ms`;
  }
}

// Market Segment helper (CCR, RCR, OCR)
// ---------------------------------------------------------------------------
// Forecast extrapolation
// ---------------------------------------------------------------------------
// Gradient-boosted trees cannot extrapolate: any sale_year beyond the training
// range lands in the same terminal leaf, so the raw model returns a flat (or
// slightly declining) price for every future year. The trainers measure the
// historical CAGR and ship it in the scaler JSON; we clamp sale_year to the last
// year the model actually saw, then apply that growth for the remaining years.

const FALLBACK_ANNUAL_GROWTH = 0.025;

// Matches MRT_WALKABLE_KM in pipeline_public/03_train_model.py.
// Mid-year: the month fed to the model when projecting, so a forecast does not
// inherit a seasonal step from the month the user happened to pick.
const NEUTRAL_FORECAST_MONTH = 6;

const MRT_WALKABLE_KM = 0.4;
// Island-wide median distance, used when a block cannot be resolved.
const MRT_FALLBACK_KM = 0.513;

// Resolved from the postal code the user typed. Falls back to the island median
// so a flat with no resolved address is priced as typical rather than as
// unusually near or far from a station.
function lookupMrtDistance() {
  const fallback = (mrtProximity && mrtProximity.median_km) || MRT_FALLBACK_KM;
  if (!mrtProximity || !mrtProximity.by_postal) return fallback;
  const postal = el.postalInput ? el.postalInput.value.trim().replace(/\D/g, '') : '';
  if (postal.length !== 6) return fallback;
  const km = mrtProximity.by_postal[postal];
  return typeof km === 'number' ? km : fallback;
}

// Returns the sale_year to feed the model, plus the multiplier to apply after.
// `segment` selects a market-specific growth rate where one is published: CCR,
// RCR and OCR have moved very differently (roughly +20%, +33% and +46% from
// 2021-Q1 to 2026-Q2), so a single blended rate overstates prime-district
// forecasts badly - about 24% too high six years out.
function getForecastAdjustment(scaler, saleYear, saleMonth, segment) {
  const cutoffYear = scaler && scaler.latest_complete_year
    ? scaler.latest_complete_year
    : (scaler && scaler.latest_year ? scaler.latest_year : saleYear);

  if (!(saleYear > cutoffYear)) {
    return { modelYear: saleYear, modelMonth: saleMonth,
             growthFactor: 1.0, yearsProjected: 0, growthRate: 0 };
  }

  const segmentGrowth = (segment && scaler && scaler.segment_growth)
    ? scaler.segment_growth[segment] : undefined;
  const growth = (typeof segmentGrowth === 'number')
    ? segmentGrowth
    : ((scaler && typeof scaler.annual_growth === 'number')
        ? scaler.annual_growth
        : FALLBACK_ANNUAL_GROWTH);

  // Time is measured from the END of the cutoff year, which is the last point
  // the model has seen. December of the cutoff year is zero years out; the
  // following January is one month out, not one year. Measuring from the START
  // of the cutoff year instead would add a spurious extra year of growth to
  // every projection.
  const month = Math.min(12, Math.max(1, saleMonth || 6));
  const yearsProjected = Math.max(0, (saleYear - cutoffYear - 1) + month / 12);

  // sale_month is a raw numeric feature, so the trees split it at thresholds
  // that mostly encode within-year price drift rather than true seasonality.
  // Feeding a projected target month through it makes December and the
  // following January differ by ~8%, which reads as a price cliff to anyone
  // stepping through the picker. Elapsed time is already carried by
  // growthFactor, so projections use a neutral mid-year month instead.
  const modelMonth = NEUTRAL_FORECAST_MONTH;

  return {
    modelYear: cutoffYear,
    modelMonth: modelMonth,
    growthFactor: Math.pow(1 + growth, yearsProjected),
    yearsProjected: yearsProjected,
    growthRate: growth
  };
}

// Districts the condo model was actually trained on. Some valid postal districts
// (notably D24 - Lim Chu Kang / Tengah) have no private resale transactions at
// all, so there is no dist_* feature for them; without a fallback every district
// flag would be zero and the model would silently price a "districtless" condo.
function resolveModelDistrict(district) {
  const padded = String(district).padStart(2, '0');
  const known = (condoScalerData && condoScalerData.districts) || null;
  if (!known || known.includes(padded)) {
    return { code: padded, substituted: false };
  }
  // Fall back to the nearest district sharing the same market segment.
  const segment = getMarketSegment(padded);
  const sameSegment = known.filter(d => getMarketSegment(d) === segment);
  const target = parseInt(padded, 10);
  let best = sameSegment.length ? sameSegment[0] : known[0];
  let bestDist = Infinity;
  for (const d of (sameSegment.length ? sameSegment : known)) {
    const gap = Math.abs(parseInt(d, 10) - target);
    if (gap < bestDist) { bestDist = gap; best = d; }
  }
  return { code: best, substituted: true, requested: padded };
}

// Market Segment helper (CCR, RCR, OCR)
// ---------------------------------------------------------------------------
// Tier 1: Project-level ground truth from URA caveats (models/condo_project_encoding.json).
//         Corrects split-district planning boundaries (e.g. D20 has both RCR in Bishan
//         and OCR in Ang Mo Kio such as Nuovo).
// Tier 2: District-level fallback when project is unlisted or left blank.
function getMarketSegment(district, projectName) {
  const norm = (projectName || '').trim().toUpperCase();
  if (norm && condoProjectEncoding && condoProjectEncoding.project_segment) {
    const seg = condoProjectEncoding.project_segment[norm];
    if (seg === 'CCR' || seg === 'RCR' || seg === 'OCR') return seg;
  }
  // Offline inlined project overrides for split districts (file:// protocol resilience)
  if (norm && typeof DEFAULT_PROJECT_SEGMENTS !== 'undefined' && DEFAULT_PROJECT_SEGMENTS[norm]) {
    return DEFAULT_PROJECT_SEGMENTS[norm];
  }
  const d = parseInt(district, 10);
  if ([1, 2, 6, 7, 9, 10, 11].includes(d)) return 'CCR';
  if ([3, 4, 8, 12, 13, 14, 15, 20, 21].includes(d)) return 'RCR';
  return 'OCR';
}

function updateDistrictSegment() {
  if (!el.condoDistrictSelect || !el.condoSegmentBadge) return;
  const dist = el.condoDistrictSelect.value;
  const projName = (el.condoProjectInput && el.condoProjectInput.value.trim()) || '';
  const segment = getMarketSegment(dist, projName);
  el.condoSegmentBadge.textContent = segment;
  el.condoSegmentBadge.className = `segment-badge segment-${segment.toLowerCase()}`;
}

// Switch Property Class (Public HDB vs Private Condo)
function setPropertyClass(pClass) {
  currentPropertyClass = pClass;

  if (pClass === 'public') {
    el.btnClassPublic.classList.add('active');
    el.btnClassPrivate.classList.remove('active');
    el.sectionPublic.style.display = 'block';
    el.sectionPrivate.style.display = 'none';

    // Adjust bounds for HDB
    el.floorAreaSlider.min = '28';
    el.floorAreaSlider.max = '215';
    el.floorAreaBounds.textContent = '28 to 215 sqm';
    if (parseFloat(el.floorAreaSlider.value) > 215) el.floorAreaSlider.value = '90';
  } else {
    el.btnClassPrivate.classList.add('active');
    el.btnClassPublic.classList.remove('active');
    el.sectionPublic.style.display = 'none';
    el.sectionPrivate.style.display = 'block';

    // Adjust bounds for Private
    el.floorAreaSlider.min = '30';
    el.floorAreaSlider.max = '350';
    el.floorAreaBounds.textContent = '30 to 350 sqm';
    if (parseFloat(el.floorAreaSlider.value) < 45) el.floorAreaSlider.value = '105';
  }

  updateFloorAreaDisplay();
  updateSaleDateModeUI(); // cutoff differs between HDB (Sep 2026) and Condo (Jun 2026)
  runPrediction();
  setMapMode(pClass);
}

// Switch Map Category (Public HDB vs Private Condo)
function setMapMode(mapType) {
  const btnPub = document.getElementById('btnMapPublic');
  const btnPriv = document.getElementById('btnMapPrivate');
  const framePub = document.getElementById('map-frame-public-wrapper');
  const framePriv = document.getElementById('map-frame-private-wrapper');
  const newTabBtn = document.getElementById('openMapNewTabBtn');
  const mapTitle = document.getElementById('mapTitle');
  const mapSubtitle = document.getElementById('mapSubtitle');

  if (!btnPub || !btnPriv) return;

  if (mapType === 'public') {
    btnPub.classList.add('active');
    btnPriv.classList.remove('active');
    if (framePub) framePub.style.display = 'block';
    if (framePriv) framePriv.style.display = 'none';
    if (newTabBtn) newTabBtn.href = 'pipeline_public/hdb_resale_price_map_clickable.html';
    if (mapTitle) mapTitle.textContent = '🗺️ HDB Public Housing Resale Map';
    if (mapSubtitle) mapSubtitle.textContent = 'Click on any block marker across Singapore to inspect historical resale transactions.';
  } else {
    btnPriv.classList.add('active');
    btnPub.classList.remove('active');
    if (framePub) framePub.style.display = 'none';
    if (framePriv) framePriv.style.display = 'block';
    if (newTabBtn) newTabBtn.href = 'pipeline_private/condo_resale_price_map_clickable.html';
    if (mapTitle) mapTitle.textContent = '🗺️ Private Condominium Resale Map';
    if (mapSubtitle) mapSubtitle.textContent = 'Explore 2,430 Private Condominiums & ECs with URA caveat resale benchmarks and CCR/RCR/OCR market segments.';
  }
}

// HDB Postal Lookup
function handleHdbPostalLookup() {
  const query = el.postalInput.value.trim().replace(/\D/g, '');
  if (query.length === 6) {
    // 1. Direct match in postal dictionary
    if (postalLookup && postalLookup[query]) {
      const match = postalLookup[query][0];
      const town = match.town;
      const leaseYear = match.lease_commence_date;

      if (town) el.townSelect.value = town;
      if (leaseYear) el.leaseYearInput.value = leaseYear;

      el.postalBadge.textContent = `✓ ${town} (${leaseYear})`;
      el.postalBadge.style.display = 'inline-block';
      runPrediction();
      return;
    }

    // 2. Fallback to sector mapping
    const sector = query.slice(0, 2);
    if (SECTOR_TO_HDB_TOWN[sector]) {
      const town = SECTOR_TO_HDB_TOWN[sector];
      el.townSelect.value = town;
      el.postalBadge.textContent = `✓ ${town}`;
      el.postalBadge.style.display = 'inline-block';
      runPrediction();
      return;
    }

    el.postalBadge.textContent = 'Postal not in database';
    el.postalBadge.style.display = 'inline-block';
  } else {
    el.postalBadge.style.display = 'none';
  }
}

// Private Condo Postal Lookup
function handleCondoPostalLookup() {
  const query = el.condoPostalInput.value.trim().replace(/\D/g, '');
  if (query.length === 6) {
    // 1. Direct match in condo database
    if (condoLookup && condoLookup[query]) {
      const match = condoLookup[query];
      if (match.project_name) el.condoProjectInput.value = match.project_name;
      currentCondoStreet = (match.street || '').trim().toUpperCase();
      if (match.property_type === 'EC' || (match.project_name && match.project_name.toLowerCase().includes('executive'))) {
        el.condoTypeSelect.value = 'EXECUTIVE CONDOMINIUM';
      } else {
        el.condoTypeSelect.value = 'CONDOMINIUM';
      }

      // Auto-detect district from postal sector
      const sector = query.slice(0, 2);
      if (SECTOR_TO_DISTRICT[sector]) {
        el.condoDistrictSelect.value = SECTOR_TO_DISTRICT[sector];
      }
      updateDistrictSegment();

      const seg = getMarketSegment(el.condoDistrictSelect.value, match.project_name);
      el.condoPostalBadge.textContent = `✓ ${match.project_name} (${seg})`;
      el.condoPostalBadge.style.display = 'inline-block';
      runPrediction();
      return;
    }

    // 2. Sector fallback (matches district and segment)
    const sector = query.slice(0, 2);
    if (SECTOR_TO_DISTRICT[sector]) {
      const dist = SECTOR_TO_DISTRICT[sector];
      el.condoDistrictSelect.value = dist;
      updateDistrictSegment();
      const seg = (typeof SECTOR_TO_SEGMENT !== 'undefined' && SECTOR_TO_SEGMENT[sector])
        ? SECTOR_TO_SEGMENT[sector]
        : getMarketSegment(dist, (el.condoProjectInput && el.condoProjectInput.value) || '');
      el.condoSegmentBadge.textContent = seg;
      el.condoSegmentBadge.className = `segment-badge segment-${seg.toLowerCase()}`;
      el.condoPostalBadge.textContent = `✓ District D${dist} (${seg})`;
      el.condoPostalBadge.style.display = 'inline-block';
      runPrediction();
      return;
    }

    el.condoPostalBadge.textContent = 'Postal not recognized';
    el.condoPostalBadge.style.display = 'inline-block';
  } else {
    el.condoPostalBadge.style.display = 'none';
  }
}

// Floor area display update
function updateFloorAreaDisplay() {
  const val = parseFloat(el.floorAreaSlider.value);
  if (currentUnit === 'sqm') {
    el.floorAreaVal.textContent = `${val} sqm (${Math.round(val * 10.7639)} sqft)`;
  } else {
    el.floorAreaVal.textContent = `${Math.round(val * 10.7639)} sqft (${val} sqm)`;
  }
}

// Month Definitions
const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
];

// Dynamically Update Month Options based on Selected Year
// Past years: all 12 months available (pure XGBoost comparable valuation).
// Current year: only months from next month onward (avoids presenting a
//   partially-observed month as a completed prediction).
// Future years: all 12 months (index-adjusted forward projection).
function updateMonthOptions(selectedYear, preferredMonth) {
  const now = new Date();
  const currentYear = now.getFullYear();
  const currentMonth = now.getMonth() + 1;

  el.saleMonthSelect.innerHTML = '';

  // For the current calendar year, only offer months that haven't happened yet
  // so the user can't select an already-registered month as a "prediction".
  // Past years and future years get all 12 months.
  const minMonth = (selectedYear === currentYear) ? Math.min(12, currentMonth + 1) : 1;

  let matched = false;
  for (let m = minMonth; m <= 12; m++) {
    const opt = document.createElement('option');
    opt.value = String(m);
    opt.textContent = MONTH_NAMES[m - 1];
    if (preferredMonth && String(m) === String(preferredMonth)) {
      opt.selected = true;
      matched = true;
    }
    el.saleMonthSelect.appendChild(opt);
  }

  if (!matched && el.saleMonthSelect.options.length > 0) {
    el.saleMonthSelect.selectedIndex = 0;
  }
}

// Update the sale-date label and hint text to reflect which engine path will
// run for the selected year:
//   • Within training cutoff (<= cutoff) -> pure XGBoost historical / comparable valuation (measured MAPE applies)
//   • Beyond cutoff (> cutoff)          -> XGBoost + published index forward projection
//
// The cutoff is read from the loaded scaler (latest_complete_year) dynamically
// for the active property class (HDB vs Private Non-Landed).
function updateSaleDateModeUI() {
  const yearLabel   = document.getElementById('saleYearLabel');
  const monthLabel  = document.getElementById('saleMonthLabel');
  const hintIcon    = document.getElementById('saleDateHintIcon');
  const hintText    = document.getElementById('saleDateHintText');
  if (!yearLabel || !hintIcon || !hintText) return;

  const selectedYear = parseInt(el.saleYearSelect.value, 10);
  const now = new Date();
  const currentYear = now.getFullYear();

  // Determine the effective model cutoff. Scaler carries latest_complete_year;
  // publishedMetrics carries forecast_base_year; fallback to DEFAULT_METRICS.
  const hdbCutoff = (scalerData && scalerData.latest_complete_year)
    || (publishedMetrics && publishedMetrics.hdb && publishedMetrics.hdb.forecast_base_year)
    || DEFAULT_METRICS.hdb.forecast_base_year;
  const condoCutoff = (condoScalerData && condoScalerData.latest_complete_year)
    || (publishedMetrics && publishedMetrics.condo && publishedMetrics.condo.forecast_base_year)
    || DEFAULT_METRICS.condo.forecast_base_year;
  const cutoff = (currentPropertyClass === 'private') ? condoCutoff : hdbCutoff;

  // Dynamic error rates updated from models/metrics.json
  const rate = (currentPropertyClass === 'private') ? CONDO_MAPE_ERROR_RATE : HDB_MAPE_ERROR_RATE;
  const label = (currentPropertyClass === 'private') ? 'Condo' : 'HDB';
  const mapeText = `${(rate * 100).toFixed(2)}% ${label}`;

  if (selectedYear <= cutoff) {
    // Mode 1: Historical / Comparable valuation — pure XGBoost, no index multiplication
    yearLabel.textContent  = 'Sale Year (Historical)';
    if (monthLabel) monthLabel.textContent = 'Sale Month (Historical)';
    hintIcon.textContent   = '📊';
    hintText.textContent   = `Historical comparable valuation — XGBoost runs directly on ${selectedYear} market data (within ${cutoff} training cutoff). Measured holdout MAPE (${mapeText}) applies.`;
    hintText.style.color   = 'var(--accent-success, #10b981)';
  } else {
    // Mode 2: Beyond cutoff — index-adjusted forward projection
    yearLabel.textContent  = 'Target Sale Year (Forecast)';
    if (monthLabel) monthLabel.textContent = 'Target Sale Month';
    hintIcon.textContent   = '🔮';
    hintText.textContent   = `Forward projection — XGBoost estimate at ${cutoff} cutoff scaled by published market index. Growth assumption compounds with horizon.`;
    hintText.style.color   = '';
  }
}

// Populate Sale Year & Month Options
// Range: from TRAINING_START_YEAR (2017, the earliest year in the HDB dataset)
// through currentYear + 6, giving users both historical back-testing and
// forward-projection capability.  The label and hint text are updated by
// updateSaleDateModeUI() to reflect which engine path will be used.
const TRAINING_START_YEAR = 2017;

function populateDateSelectors() {
  const now = new Date();
  const currentYear = now.getFullYear();
  const currentMonth = now.getMonth() + 1;

  const hdbCutoff = (scalerData && scalerData.latest_complete_year)
    || (publishedMetrics && publishedMetrics.hdb && publishedMetrics.hdb.forecast_base_year)
    || DEFAULT_METRICS.hdb.forecast_base_year;
  const condoCutoff = (condoScalerData && condoScalerData.latest_complete_year)
    || (publishedMetrics && publishedMetrics.condo && publishedMetrics.condo.forecast_base_year)
    || DEFAULT_METRICS.condo.forecast_base_year;
  const cutoff = (currentPropertyClass === 'private') ? condoCutoff : hdbCutoff;

  // Build the full range: historical years first, then forecast
  el.saleYearSelect.innerHTML = '';
  for (let yr = TRAINING_START_YEAR; yr <= currentYear + 6; yr++) {
    const opt = document.createElement('option');
    opt.value = yr;
    if (yr <= cutoff) {
      opt.textContent = `${yr} (historical)`;
    } else {
      opt.textContent = `${yr} (forecast)`;
    }
    el.saleYearSelect.appendChild(opt);
  }

  // Default: next upcoming future month so the page opens in forward-projection
  // mode, which is the primary use case (selling/buying planning).
  let targetYear = currentYear;
  let targetMonth = currentMonth + 1;
  if (targetMonth > 12) {
    targetYear = currentYear + 1;
    targetMonth = 1;
  }

  el.saleYearSelect.value = String(targetYear);
  updateMonthOptions(targetYear, targetMonth);
  updateSaleDateModeUI();
}

// Event Listeners
function setupEventListeners() {
  // Property Class Toggle
  el.btnClassPublic.addEventListener('click', () => setPropertyClass('public'));
  el.btnClassPrivate.addEventListener('click', () => setPropertyClass('private'));

  // HDB Postal Lookup & Dropdowns
  el.postalInput.addEventListener('input', handleHdbPostalLookup);
  [el.townSelect, el.flatTypeSelect, el.flatModelSelect, el.saleMonthSelect].forEach(input => {
    input.addEventListener('change', runPrediction);
  });
  el.saleYearSelect.addEventListener('change', () => {
    const yr = parseInt(el.saleYearSelect.value, 10);
    const curMonth = el.saleMonthSelect.value;
    updateMonthOptions(yr, curMonth);
    updateSaleDateModeUI(); // swap label + hint between Historical / Forecast
    runPrediction();
  });
  el.leaseYearInput.addEventListener('input', runPrediction);

  // Private Condo Inputs
  el.condoPostalInput.addEventListener('input', handleCondoPostalLookup);
  el.condoProjectInput.addEventListener('input', () => {
    updateDistrictSegment();
    runPrediction();
  });
  el.condoTypeSelect.addEventListener('change', runPrediction);
  el.condoDistrictSelect.addEventListener('change', () => {
    updateDistrictSegment();
    runPrediction();
  });
  el.condoTenureSelect.addEventListener('change', (e) => {
    if (e.target.value === '99') {
      el.condoLeaseGroup.style.display = 'block';
    } else {
      el.condoLeaseGroup.style.display = 'none';
    }
    runPrediction();
  });
  el.condoLeaseYearInput.addEventListener('input', runPrediction);

  // Storey slider
  el.storeySlider.addEventListener('input', (e) => {
    el.storeyVal.textContent = `Flr ${e.target.value}`;
    runPrediction();
  });

  // Floor area slider
  el.floorAreaSlider.addEventListener('input', () => {
    updateFloorAreaDisplay();
    runPrediction();
  });

  // Unit Toggle (sqm <-> sqft)
  el.sqmBtn.addEventListener('click', () => setUnit('sqm'));
  el.sqftBtn.addEventListener('click', () => setUnit('sqft'));

  // Theme Toggle
  el.themeToggleBtn.addEventListener('click', toggleTheme);

  // Tab Navigation
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      const tabTarget = btn.getAttribute('data-tab');
      btn.classList.add('active');
      const targetEl = document.getElementById(tabTarget);
      if (targetEl) targetEl.classList.add('active');
    });
  });

  // Map Category Toggle Listeners
  const btnMapPub = document.getElementById('btnMapPublic');
  const btnMapPriv = document.getElementById('btnMapPrivate');
  if (btnMapPub) btnMapPub.addEventListener('click', () => setMapMode('public'));
  if (btnMapPriv) btnMapPriv.addEventListener('click', () => setMapMode('private'));
}

// Unit Switching (sqm <-> sqft)
function setUnit(unit) {
  currentUnit = unit;
  if (unit === 'sqm') {
    el.sqmBtn.classList.add('active');
    el.sqftBtn.classList.remove('active');
  } else {
    el.sqftBtn.classList.add('active');
    el.sqmBtn.classList.remove('active');
  }
  updateFloorAreaDisplay();
}

// Theme Management
function initTheme() {
  const savedTheme = localStorage.getItem('hdb_theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('hdb_theme', next);
  updateThemeIcon(next);
}

function updateThemeIcon(theme) {
  el.themeToggleBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
}

function updateStatus(text, status) {
  if (el.engineStatus) {
    el.engineStatus.innerHTML = `<span class="status-dot"></span>${text}`;
  }
}

function formatNumber(num) {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// ---------------------------------------------------------------------------
// Published metrics
// ---------------------------------------------------------------------------
// The trainers write models/metrics.json on every run and the About panel reads
// it here, so the advertised accuracy always describes the model actually
// shipped. Nothing in index.html needs editing after a retrain.

function setMetricText(key, value) {
  document.querySelectorAll(`[data-metric="${key}"]`).forEach(node => {
    node.textContent = value;
  });
}

function renderPublishedMetrics(metrics) {
  if (!metrics) return;

  ['hdb', 'condo'].forEach(key => {
    const m = metrics[key];
    if (!m) return;

    if (typeof m.r2 === 'number') {
      setMetricText(`${key}.r2`, m.r2.toFixed(4));
      setMetricText(`${key}.r2_label`, `R² Score (${(m.r2 * 100).toFixed(1)}%)`);
    }
    if (typeof m.mae === 'number') {
      setMetricText(`${key}.mae`, `S$ ${formatNumber(Math.round(m.mae))}`);
    }
    if (typeof m.trees === 'number') {
      setMetricText(`${key}.trees`, `${formatNumber(m.trees)} Trees`);
    }
    if (typeof m.records === 'number') {
      setMetricText(`${key}.records`, formatNumber(m.records));
    }
    if (typeof m.mape === 'number') {
      const holdout = m.holdout_samples ? `${formatNumber(m.holdout_samples)} ` : '';
      setMetricText(`${key}.mae_note`,
        `Typical error ${m.mape.toFixed(2)}% across ${holdout}held-out future sales. ` +
        `Measures valuation, not the forward projection.`);
    }
    if (m.validation) {
      setMetricText(`${key}.r2_note`, `Measured on a ${m.validation} holdout.`);
    }
    // Forecast basis: the growth rate applied past the training cutoff, and
    // where it came from. Shown so the projection's assumption is visible
    // rather than buried in the model.
    if (m.segment_growth && Object.keys(m.segment_growth).length) {
      const parts = ['CCR', 'RCR', 'OCR']
        .filter(seg => typeof m.segment_growth[seg] === 'number')
        .map(seg => `${seg} ${(m.segment_growth[seg] * 100).toFixed(1)}%`);
      setMetricText(`${key}.growth`, parts.length ? parts.join(' / ') : '--');
    } else if (typeof m.annual_growth === 'number') {
      setMetricText(`${key}.growth`, `${(m.annual_growth * 100).toFixed(2)}% / yr`);
    }
    if (m.index_source) {
      const base = m.forecast_base_year ? ` ${m.forecast_base_year}` : ' the training cutoff';
      setMetricText(`${key}.growth_note`,
        `Assumption, not a measurement. Dates beyond${base} are scaled by the ` +
        `${m.index_source}.`);
    } else if (typeof m.annual_growth === 'number') {
      setMetricText(`${key}.growth_note`,
        'Trend fitted from transactions; official index unavailable at build time.');
    }

    if (typeof m.features === 'number') {
      const extra = key === 'condo' && m.projects
        ? ` across ${formatNumber(m.projects)} named developments`
        : '';
      setMetricText(`${key}.features_note`, `${m.features} engineered features${extra}.`);
    }
  });

  // Confidence bands are the honest forward-looking MAPE, so the range shown
  // beside a prediction matches the error the model actually makes.
  const hdbMape = metrics.hdb && metrics.hdb.mape;
  const condoMape = metrics.condo && metrics.condo.mape;
  if (typeof hdbMape === 'number') HDB_MAPE_ERROR_RATE = hdbMape / 100;
  if (typeof condoMape === 'number') CONDO_MAPE_ERROR_RATE = condoMape / 100;
  if (typeof hdbMape === 'number' && typeof condoMape === 'number') {
    setMetricText('bands', `±${hdbMape.toFixed(1)}% / ±${condoMape.toFixed(1)}%`);
  }
  updateSaleDateModeUI();
  runPrediction();
}

// Feeds the condo project autocomplete. Only projects the model was trained on
// are offered, so a chosen name always resolves to a real encoding.
function populateProjectDatalist(encoding) {
  const list = document.getElementById('condoProjectList');
  if (!list || !encoding || !encoding.project_psf) return;
  const names = Object.keys(encoding.project_psf).sort();
  const frag = document.createDocumentFragment();
  names.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    frag.appendChild(opt);
  });
  list.innerHTML = '';
  list.appendChild(frag);
}

// Initialize Application
async function initApp() {
  initElements();
  initTheme();
  populateDateSelectors();
  setupEventListeners();
  updateDistrictSegment();

  // Run initial calculation immediately in 0 ms so screen is instantly active
  runPrediction();

  // If opened as local file (file:/// protocol), browser blocks fetch() via CORS.
  // The engine is designed to run seamlessly with built-in baseline models.
  const isFileProtocol = window.location.protocol === 'file:';
  if (isFileProtocol) {
    // fetch() is blocked on file://, so render the figures compiled into the
    // bundle rather than leaving every metric card showing its placeholder.
    renderPublishedMetrics(DEFAULT_METRICS);
    updateStatus('Local File Mode active • In-browser valuation engines ready', 'ready');
    return;
  }

  updateStatus('Loading ML model & address assets in background...', 'loading');

  // Helper to fetch with path fallbacks
  const fetchWithFallback = (paths) => {
    return paths.reduce((chain, path) => {
      return chain.catch(() => fetch(path).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      }));
    }, Promise.reject());
  };

  // Asynchronously load scaler
  fetchWithFallback(['models/hdb_scaler.json'])
    .then(data => { scalerData = data; })
    .catch(() => { /* Uses DEFAULT_SCALER */ });

  // Asynchronously load HDB 600-tree XGBoost model
  fetchWithFallback(['models/hdb_model_web.json'])
    .then(data => {
      modelData = data;
      isXgbReady = true;
      const totalPostals = Object.keys(postalLookup).length + Object.keys(condoLookup).length;
      updateStatus(`Engine ready (HDB + URA Condo ensembles, ${totalPostals > 0 ? totalPostals.toLocaleString() : '19.9k'} postals)`, 'ready');
      runPrediction();
    })
    .catch(err => {
      console.warn('XGBoost model file not loaded, running calibrated statistical baseline:', err.message);
      updateStatus('Engine ready (Statistical ML + URA Condo Hedonic)', 'ready');
    });

  // Published accuracy figures, written by the trainers. Rendered into the page
  // so the About panel can never drift from the model actually shipped.
  fetchWithFallback(['models/metrics.json'])
    .then(data => { publishedMetrics = data; renderPublishedMetrics(data); })
    .catch(() => { /* Panel keeps its build-time defaults. */ });

  // MRT proximity lookup for HDB transport features.
  fetchWithFallback(['pipeline_public/data/hdb_mrt_proximity.json'])
    .then(data => { mrtProximity = data; runPrediction(); })
    .catch(() => { /* Falls back to the island median distance. */ });

  // Project-level price encodings for the condo model.
  fetchWithFallback(['models/condo_project_encoding.json'])
    .then(data => { condoProjectEncoding = data; populateProjectDatalist(data); updateDistrictSegment(); runPrediction(); })
    .catch(() => { /* Model falls back to the global mean for project features. */ });

  // Asynchronously load Private Condo scaler & XGBoost model
  fetchWithFallback(['models/condo_scaler.json'])
    .then(data => { condoScalerData = data; updateSaleDateModeUI(); })
    .catch(() => { /* Uses hedonic fallback */ });

  fetchWithFallback(['models/condo_model_web.json'])
    .then(data => {
      condoModelData = data;
      isCondoXgbReady = true;
      updateStatus('Engine ready (600 HDB Trees + 285 URA Condo Trees)', 'ready');
      if (currentPropertyClass === 'private') runPrediction();
    })
    .catch(() => { /* Uses hedonic fallback */ });

  // Asynchronously load HDB postal dictionary
  fetchWithFallback([
    'pipeline_public/data/hdb_postal_data.json',
    'hdb_postal_data.json',
    'postal_data.json'
  ])
    .then(data => { postalLookup = data || {}; })
    .catch(() => { /* Fallback to sector mapping */ });

  // Asynchronously load Private Condo postal dictionary
  fetchWithFallback([
    'pipeline_private/data/private_nonlanded_postal_data.json',
    // Previous filename, kept so a deployment that has not picked up the
    // rename yet still resolves.
    'pipeline_private/data/condo_postal_data.json',
    'condo_postal_data.json'
  ])
    .then(data => { condoLookup = data || ((typeof DEFAULT_CONDO_LOOKUP !== 'undefined') ? DEFAULT_CONDO_LOOKUP : {}); })
    .catch(() => { /* Fallback to DEFAULT_CONDO_LOOKUP and sector mapping */ });
}

// Bootstrap
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
