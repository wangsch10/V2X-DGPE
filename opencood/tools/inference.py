"""
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
"""
from argparse import Namespace
import umap.umap_ as umap
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')
val_list=["000211", "000212", "000214", "000215", "000216", "000217", "000871", "000872", "000873", "000874", "000876", "000881", "000883", "000898", "000903", "000904", "000905", "000907", "000909", "000911", "000916", "000917", "000918", "000919", "000920", "000921", "000922", "000923", "000924", "000925", "000927", "000928", "000929", "000930", "000931", "000932", "000933", "000934", "000935", "000936", "000937", "000938", "000939", "000940", "000941", "000942", "000943", "000944", "000945", "000946", "000947", "000948", "000950", "000951", "000952", "000953", "000954", "000955", "000956", "000957", "000958", "000959", "000960", "000962", "000963", "000964", "000965", "000971", "000973", "000974", "000978", "000979", "000988", "000990", "000991", "000992", "000993", "000996", "000997", "000998", "000999", "001000", "001002", "001003", "001005", "001006", "001008", "001009", "001010", "001011", "001012", "001013", "001014", "001015", "001016", "001017", "001018", "001019", "001020", "001021", "001022", "001023", "001024", "001025", "001026", "001027", "001028", "001029", "001030", "001031", "001032", "001033", "001034", "001035", "001036", "001037", "001038", "001039", "001040", "001041", "001043", "001044", "001047", "001048", "001054", "001070", "001071", "001073", "001074", "001075", "001076", "001077", "001718", "001719", "001741", "001745", "001746", "001747", "001748", "001749", "001750", "001751", "001752", "001753", "001754", "001755", "001756", "001757", "001758", "001759", "001760", "001761", "001762", "001763", "001764", "001766", "001767", "001768", "001769", "001770", "001771", "001772", "001773", "001774", "001775", "001777", "001778", "001779", "001780", "001781", "001782", "001783", "001785", "001786", "001787", "001788", "001789", "001791", "001793", "001794", "001795", "001819", "001822", "001823", "001824", "001825", "001826", "001827", "001828", "001829", "001830", "001831", "001832", "001833", "001834", "001835", "001836", "001837", "001838", "001839", "001840", "001841", "001842", "001843", "001844", "001845", "001846", "001847", "001848", "001849", "001850", "001851", "001852", "001853", "001854", "001855", "001856", "001857", "001858", "001859", "001860", "001861", "003068", "003069", "003070", "003071", "003072", "003073", "003074", "003075", "003076", "003077", "003078", "003079", "003080", "003081", "003125", "003127", "003128", "003129", "003130", "003131", "003132", "003133", "003134", "003135", "003136", "003137", "003138", "003139", "003140", "003141", "003142", "003143", "003144", "003145", "003146", "003147", "003148", "003149", "003150", "003151", "003152", "003153", "003154", "003155", "003156", "003157", "003158", "003159", "003160", "003161", "003162", "003163", "003164", "003165", "003166", "003167", "003168", "003169", "003170", "003171", "003172", "003173", "003174", "003175", "003176", "003177", "003178", "003179", "003180", "003181", "003182", "003183", "003184", "003185", "003186", "003187", "003188", "003189", "003190", "003191", "003192", "003233", "003234", "003235", "003236", "003237", "003238", "003239", "003240", "003241", "003242", "003243", "003244", "003245", "003246", "003247", "003248", "003249", "003250", "003251", "003252", "003253", "003254", "003255", "003256", "003257", "003258", "003259", "003260", "003261", "003262", "003263", "003264", "003265", "003266", "003267", "003268", "003269", "003270", "003271", "003272", "003273", "003316", "003317", "003318", "003319", "003320", "003321", "003322", "003323", "003324", "003325", "003326", "003327", "003328", "003329", "003330", "003331", "003332", "003333", "003334", "003335", "003336", "003337", "003338", "003339", "003340", "003341", "003342", "003343", "003344", "003345", "003346", "003347", "003348", "003349", "003350", "003351", "003352", "003353", "003354", "003355", "003356", "003357", "003358", "003359", "003360", "003361", "003362", "003363", "003364", "003365", "003366", "003367", "003368", "003369", "003370", "003371", "003372", "003373", "003374", "003375", "003376", "003377", "003378", "003379", "003380", "003392", "003569", "003570", "003571", "003572", "003573", "003574", "003575", "003576", "003577", "003579", "003580", "003581", "003582", "003583", "003584", "003585", "003586", "003587", "003588", "003589", "003608", "003609", "003610", "003611", "003612", "003613", "003614", "003615", "003616", "003617", "003618", "003619", "003620", "003621", "003622", "003623", "003624", "003625", "003626", "003627", "003628", "003629", "003630", "003631", "003633", "003641", "003676", "003677", "003678", "003679", "004074", "004075", "004076", "004077", "004078", "004079", "004080", "004081", "004082", "004083", "004084", "004085", "004086", "004087", "004088", "004089", "004090", "004091", "004092", "004093", "004094", "004095", "004096", "004097", "004098", "004099", "004100", "004101", "004102", "004103", "004104", "004105", "004106", "004107", "004108", "004109", "004110", "004111", "004112", "004113", "004114", "004115", "004116", "004117", "004118", "004119", "004120", "004121", "004122", "004123", "004124", "004125", "004126", "004127", "004128", "004129", "004130", "004131", "004132", "004133", "004134", "004135", "004180", "004182", "004183", "004185", "004186", "004187", "004188", "004189", "004190", "004191", "004192", "004193", "004194", "004195", "004196", "004197", "004198", "004199", "004200", "004201", "004202", "004203", "004204", "004205", "004206", "004207", "004209", "004210", "004211", "004212", "004213", "004214", "004215", "004216", "004217", "004218", "004219", "004220", "004221", "004222", "004223", "004224", "004225", "004226", "004227", "004228", "004229", "004230", "004231", "004232", "004233", "004234", "004235", "004236", "004237", "004238", "004239", "004241", "004242", "004243", "004261", "004262", "004263", "004264", "004265", "004266", "004267", "004268", "004269", "004270", "004271", "004272", "004273", "004274", "004275", "004276", "004277", "004278", "004279", "004280", "004281", "004285", "004325", "004326", "004327", "004328", "004329", "004330", "004331", "004332", "004333", "004334", "004335", "004336", "004337", "004338", "004339", "004340", "004341", "004342", "004343", "004344", "004345", "004346", "004347", "004348", "004349", "004350", "004351", "004352", "004353", "004354", "004355", "004356", "004357", "004358", "004359", "004360", "004361", "004362", "004363", "004364", "004365", "004366", "004367", "004368", "004369", "004370", "004371", "004372", "004373", "004374", "004375", "004376", "004377", "004378", "004379", "004380", "004381", "004382", "004383", "004384", "004385", "004386", "004387", "004388", "004389", "004390", "004392", "004431", "004432", "004433", "004434", "004436", "004437", "004438", "004439", "004440", "004441", "004442", "004443", "004444", "004445", "004446", "004447", "004448", "004449", "007823", "007824", "007832", "009701", "010801", "010802", "010803", "010804", "010805", "010806", "010807", "010808", "010809", "010810", "010811", "010812", "010813", "010814", "010815", "010816", "010817", "010818", "010819", "010820", "010821", "010822", "010823", "010824", "010825", "010826", "010827", "010828", "010829", "010830", "010831", "010832", "010833", "010834", "010835", "010836", "010837", "010838", "010839", "010840", "010841", "010842", "010843", "010844", "010845", "010846", "010847", "010848", "010849", "010850", "010851", "010852", "010853", "010854", "010855", "010856", "010857", "010858", "010859", "010860", "010861", "010862", "010863", "010864", "010865", "010866", "010867", "010868", "010869", "010870", "010871", "010872", "010873", "010874", "010875", "010876", "010877", "010878", "010879", "010880", "010881", "010882", "010883", "010884", "010885", "010886", "010887", "010888", "010889", "010890", "010891", "010892", "010893", "010894", "010895", "010896", "010897", "010898", "010899", "010900", "010901", "010902", "010903", "010904", "010905", "010906", "010907", "010908", "010909", "010910", "010911", "010912", "010913", "010914", "010915", "010916", "010917", "010918", "010919", "010920", "010921", "010922", "010923", "010924", "010925", "010926", "010927", "010928", "010929", "010930", "010931", "010932", "010933", "010934", "010935", "010936", "010937", "010938", "010939", "010940", "010941", "010942", "010943", "010944", "010945", "010946", "010947", "010948", "010949", "010950", "010951", "010952", "010953", "010954", "010955", "010956", "010957", "010958", "010959", "010960", "010961", "010962", "010963", "010964", "010965", "010966", "010967", "010968", "010969", "010970", "010971", "010972", "010973", "010974", "010975", "010976", "010977", "010978", "010979", "010980", "010981", "010982", "010983", "010984", "010985", "010986", "010987", "010988", "010989", "010990", "010991", "010992", "010993", "010994", "010995", "010996", "010997", "010998", "010999", "011000", "011001", "011002", "011003", "011004", "011005", "011006", "011007", "011008", "011009", "011010", "011011", "011012", "011013", "011014", "011015", "011016", "011017", "011018", "011019", "011020", "011021", "011022", "011023", "011024", "011025", "011026", "011027", "011028", "011029", "011030", "011031", "011032", "011033", "011034", "011035", "011036", "011037", "011038", "011039", "011040", "011041", "011042", "011043", "011044", "011045", "011046", "011047", "011048", "011049", "011181", "011194", "011196", "011203", "011226", "011235", "011242", "011244", "011246", "011255", "011494", "011495", "011496", "011497", "011498", "011499", "011500", "011501", "011502", "011503", "011505", "011509", "011512", "011527", "011530", "011533", "011534", "011535", "011539", "011542", "011544", "011545", "011548", "011549", "011550", "011551", "011552", "011553", "011555", "011556", "011557", "011558", "011559", "011562", "011563", "011564", "011565", "011566", "011568", "011572", "011796", "011797", "011798", "011799", "011800", "011801", "011802", "011803", "011804", "011805", "011806", "011807", "011808", "011809", "011810", "011811", "011812", "011813", "011814", "011815", "011816", "011817", "011818", "011819", "011820", "011821", "011822", "011823", "011824", "011825", "011826", "011827", "011829", "011830", "011832", "011855", "011858", "011861", "011862", "011863", "011864", "011865", "011866", "011867", "011868", "011869", "011870", "011871", "011872", "011873", "011874", "011875", "011876", "011877", "011878", "011879", "011880", "011881", "011882", "011883", "011884", "011885", "011886", "011887", "011888", "011889", "011890", "011891", "011892", "011893", "011894", "011895", "011896", "011898", "011899", "011900", "011903", "011905", "011909", "011917", "011921", "011924", "011926", "011927", "011928", "011930", "011931", "011932", "011934", "011935", "011936", "011937", "011938", "011939", "011940", "011941", "011943", "011944", "011945", "011946", "011947", "011948", "011949", "011950", "011951", "011952", "011953", "011954", "011955", "011956", "011958", "011959", "013724", "013725", "013728", "013743", "013745", "013746", "013747", "013749", "013750", "013751", "013753", "013754", "013755", "013756", "013757", "013758", "013759", "013760", "013761", "013762", "013763", "013764", "013765", "013766", "013767", "013768", "013769", "013770", "013771", "013772", "013773", "013774", "013775", "013776", "013777", "013778", "013779", "013780", "013781", "013782", "013785", "013804", "013806", "013807", "013808", "013809", "013810", "013811", "013812", "013813", "013814", "013815", "013816", "013817", "013818", "013819", "013820", "013821", "013822", "013823", "013824", "013825", "013826", "013827", "013828", "013829", "013830", "013831", "013832", "013833", "013834", "013835", "013836", "013837", "013839", "013841", "013842", "013843", "013844", "013846", "013847", "013848", "013864", "013865", "013866", "013867", "013868", "013869", "013870", "013871", "013872", "013873", "013874", "013875", "013876", "013877", "013878", "013879", "013880", "013881", "013883", "013884", "013885", "013886", "013887", "013888", "013889", "013890", "013891", "013892", "013893", "013894", "013895", "013896", "013897", "013898", "013922", "013925", "014214", "014215", "014216", "014217", "014218", "014219", "014220", "014223", "014239", "014240", "014241", "014242", "014243", "014244", "014245", "014246", "014247", "014248", "014250", "014251", "014252", "014253", "014254", "014255", "014256", "014257", "014258", "014259", "014260", "014261", "014262", "014263", "014264", "014265", "014266", "014267", "014268", "014269", "014270", "014271", "014272", "014273", "014274", "014275", "014276", "014278", "014279", "014280", "014282", "014289", "014297", "014299", "014300", "014301", "014302", "014303", "014304", "014305", "014306", "014308", "014309", "014310", "014311", "014312", "014313", "014314", "014315", "014316", "014317", "014318", "014319", "014320", "014321", "014322", "014323", "014324", "014330", "014331", "014332", "014334", "014335", "014336", "014337", "014338", "014341", "014342", "014343", "014356", "014359", "014360", "014361", "014362", "014363", "014364", "014365", "014366", "014367", "014368", "014369", "014370", "014371", "014372", "014373", "014374", "014375", "014376", "014377", "014378", "014379", "014380", "014381", "014382", "014383", "014384", "014385", "014386", "014387", "014388", "014389", "014390", "014391", "014392", "014394", "014396", "014505", "014506", "014507", "014508", "014509", "014510", "014511", "014513", "014523", "014524", "014525", "014526", "014527", "014528", "014529", "014530", "014531", "014532", "014533", "014534", "014535", "014536", "014537", "014538", "014539", "017267", "017268", "017269", "017270", "017271", "017272", "017273", "017284", "017285", "017286", "017287", "017288", "017289", "017290", "017291", "017292", "017293", "017294", "017295", "017296", "017297", "017298", "017299", "017312", "017313", "017314", "017315", "017316", "017317", "017318", "017319", "017320", "017321", "017322", "017323", "017324", "017325", "017326", "017327", "017337", "017338", "017339", "017340", "017341", "017342", "017343", "017344", "017345", "017346", "017347", "017348", "017349", "017350", "017351", "017352", "017363", "017365", "017366", "017367", "017368", "017369", "017370", "017371", "017372", "017373", "017374", "017375", "017376", "017377", "017378", "017380", "017391", "017392", "017393", "017394", "017395", "017396", "017397", "017398", "017399", "017400", "017401", "017402", "017403", "017404", "017405", "017415", "017417", "017418", "017419", "017420", "017421", "017422", "017423", "017424", "017425", "017426", "017427", "017428", "017429", "017430", "017431", "017432", "017433", "017741", "017742", "017743", "017744", "017745", "017746", "017747", "017748", "017749", "017750", "017751", "017752", "017753", "017754", "017755", "017757", "017767", "017768", "017769", "017770", "017771", "017772", "017773", "017774", "017775", "017776", "017777", "017778", "017779", "017780", "017781", "017782", "017793", "017794", "018226", "018227", "018228", "018229", "018230", "018231", "018232", "018233", "018234", "018235", "018236", "018237", "018238", "018239", "018240", "018241", "018252", "018253", "018254", "018886", "018887", "018888", "018889", "018890", "018891", "018892", "018893", "018894", "018895", "018930", "018933", "018934", "018935", "018937", "018938", "018940", "018941", "018942", "018943", "018944", "018945", "018946", "018947", "018948", "018949", "018950", "018951", "018952", "018953", "018954", "018955", "018956", "018957", "018958", "018959", "018960", "018961", "018962", "018963", "018964", "018965", "018966", "018967", "018968", "018969", "018970", "018971", "018972", "018973", "018974", "018975", "018976", "018977", "018978", "018979", "018980", "018981", "018982", "018983", "018984", "018985", "018986", "018987", "018988", "018989", "018990", "018992", "018993", "018994", "019034", "019035", "019036", "019037", "019039", "019040", "019042", "019043", "019044", "019045", "019046", "019047", "019048", "019049", "019050", "019051", "019052", "019053", "019054", "019055", "019056", "019057", "019058", "019059", "019060", "019061", "019062", "019063", "019064", "019065", "019066", "019067", "019068", "019069", "019070", "019071", "019072", "019073", "019074", "019075", "019076", "019077", "019078", "019079", "019080", "019081", "019082", "019083", "019084", "019766", "019767", "019768", "019769", "019770", "019771", "019772", "019773", "019774", "019775", "019776", "019777", "019778", "019779", "019780", "019781", "019782", "019783", "019785", "019786", "019789", "019823", "019826", "019828", "019829", "019830", "019831", "019832", "019833", "019834", "019835", "019836", "019837", "019838", "019839", "019840", "019841", "019842", "019843", "019844", "019845", "019846", "019847", "019848", "019849", "019850", "019851", "019852", "019853", "019854", "019855", "019856", "019857", "019858", "019859", "019860", "019861", "019862", "019863", "019864", "019865", "019866", "019867", "019868", "019869", "019870", "019871", "019872", "019873", "019874", "019875", "019876", "019881", "019882", "019883", "019884", "019885", "019886", "019887", "019889", "019891", "019892", "019893", "019928", "019929", "019930", "019931", "019932", "019933", "019934", "019935", "019936", "019937", "019938", "019939", "019940", "019941", "019942", "019943", "019944", "019945", "019946", "019947", "019948", "019949", "019950", "019951", "019952", "019953", "019954", "019955", "019956", "019957", "019958", "019959", "019960", "019961", "019962", "019963", "019964", "019965", "019966", "019967", "019968", "019969", "019970", "019971", "019972", "019973", "019974", "019975", "019976", "019977", "019978", "019979", "019980", "019981", "019986", "019987", "019988", "019990", "019992"]
segment_dict = {
        **{frame: (0, 209) for frame in range(0, 210)},
        **{frame: (210, 217) for frame in range(210, 218)},
        **{frame: (218, 409) for frame in range(218, 410)},
        **{frame: (410, 418) for frame in range(410, 419)},
        **{frame: (419, 868) for frame in range(419, 869)},
        **{frame: (869, 870) for frame in range(869, 871)},
        **{frame: (871, 1077) for frame in range(871, 1078)},
        **{frame: (1078, 1296) for frame in range(1078, 1297)},
        **{frame: (1297, 1515) for frame in range(1297, 1516)},
        **{frame: (1715, 1864) for frame in range(1715, 1865)},
        **{frame: (1865, 1933) for frame in range(1865, 1934)},
        **{frame: (1934, 2064) for frame in range(1934, 2065)},
        **{frame: (2234, 2432) for frame in range(2234, 2433)},
        **{frame: (2433, 2571) for frame in range(2433, 2572)},
        **{frame: (3057, 3230) for frame in range(3057, 3231)},
        **{frame: (3231, 3399) for frame in range(3231, 3400)},
        **{frame: (3400, 3559) for frame in range(3400, 3560)},
        **{frame: (3560, 3679) for frame in range(3560, 3680)},
        **{frame: (3680, 3829) for frame in range(3680, 3830)},
        **{frame: (4050, 4259) for frame in range(4050, 4260)},
        **{frame: (4260, 4449) for frame in range(4260, 4450)},
        **{frame: (4450, 4582) for frame in range(4450, 4583)},
        **{frame: (4583, 4742) for frame in range(4583, 4743)},
        **{frame: (4883, 5073) for frame in range(4883, 5074)},
        **{frame: (5603, 5782) for frame in range(5603, 5783)},
        **{frame: (5783, 5932) for frame in range(5783, 5933)},
        **{frame: (6106, 6245) for frame in range(6106, 6246)},
        **{frame: (6246, 6435) for frame in range(6246, 6436)},
        **{frame: (6436, 6585) for frame in range(6436, 6586)},
        **{frame: (6586, 6705) for frame in range(6586, 6706)},
        **{frame: (6706, 6885) for frame in range(6706, 6886)},
        **{frame: (6886, 7005) for frame in range(6886, 7006)},
        **{frame: (7445, 7644) for frame in range(7445, 7645)},
        **{frame: (7645, 7814) for frame in range(7645, 7815)},
        **{frame: (7815, 7984) for frame in range(7815, 7985)},
        **{frame: (8665, 8794) for frame in range(8665, 8795)},
        **{frame: (8795, 8934) for frame in range(8795, 8935)},
        **{frame: (8935, 9109) for frame in range(8935, 9110)},
        **{frame: (9110, 9299) for frame in range(9110, 9300)},
        **{frame: (9300, 9469) for frame in range(9300, 9470)},
        **{frame: (9680, 9819) for frame in range(9680, 9820)},
        **{frame: (10010, 10219) for frame in range(10010, 10220)},
        **{frame: (10220, 10419) for frame in range(10220, 10420)},
        **{frame: (10420, 10609) for frame in range(10420, 10610)},
        **{frame: (10610, 10799) for frame in range(10610, 10800)},
        **{frame: (10800, 11049) for frame in range(10800, 11050)},
        **{frame: (11050, 11259) for frame in range(11050, 11260)},
        **{frame: (11260, 11469) for frame in range(11260, 11470)},
        **{frame: (11470, 12609) for frame in range(11470, 12610)},
        **{frame: (12610, 13926) for frame in range(12610, 13927)},
        **{frame: (13927, 14396) for frame in range(13927, 14397)},
        **{frame: (14445, 16084) for frame in range(14445, 16085)},
        **{frame: (16085, 16224) for frame in range(16085, 16225)},
        **{frame: (16225, 18054) for frame in range(16225, 18055)},
        **{frame: (18055, 18094) for frame in range(18055, 18095)},
        **{frame: (18095, 18414) for frame in range(18095, 18415)},
        **{frame: (18415, 18654) for frame in range(18415, 18655)},
        **{frame: (18885, 19084) for frame in range(18885, 19085)},
        **{frame: (19515, 20514) for frame in range(19515, 20515)},
    }
def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--model2_dir', type=str, default=None,
                        help='Continued training path for inf model')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt

def is_in_same_segment(x, history):
    segment = segment_dict.get(int(x), (None, None))
    start, end = segment
    return start <= int(history) <= end
def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single', 'late_2model', 'no_inf'] 
    print(opt.fusion_method)
    hypes = yaml_utils.load_yaml(None, opt)
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model') 
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if opt.fusion_method == 'late_2model':
        model_inf = train_utils.create_model(hypes)
        saved_path = opt.model2_dir
        resume_epoch, model_inf = train_utils.load_saved_model(saved_path, model_inf)
        print(f"inf model resume from {resume_epoch} epoch.")
        if torch.cuda.is_available():
            model_inf.cuda()
        model_inf.eval()
    # setting noise
    # np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_validate_dataset = build_dataset(hypes, visualize=True, train=False)
    delay=1
    if delay==1:
        opencood_validate_dataset.change_dataset(0,0)

        #print('((((((((((((((((((((((', opencood_train_dataset[0]['ego']['sample_idx']) #0
        for i in range(1,len(val_list)):#len(val_list)
            # time1=time.time()
            result_bool = is_in_same_segment(val_list[i], val_list[i-1])
            if result_bool:
                opencood_validate_dataset.change_dataset(i,1)
            else:
                opencood_validate_dataset.change_dataset(i,0)
###############################################历史第二帧###############################################
    elif delay==2:
        opencood_validate_dataset.change_dataset(0,0)
        opencood_validate_dataset.change_dataset(1,0)
        #print('((((((((((((((((((((((', opencood_train_dataset[0]['ego']['sample_idx']) #0
        for i in range(2,len(val_list)):#len(val_list)
            # time1=time.time()
            result_bool = is_in_same_segment(val_list[i], val_list[i-2])
            if result_bool:
                opencood_validate_dataset.change_dataset(i,1)
            else:
                opencood_validate_dataset.change_dataset(i,0)
    elif delay==3:
        opencood_validate_dataset.change_dataset(0,0)
        opencood_validate_dataset.change_dataset(1,0)
        opencood_validate_dataset.change_dataset(2,0)
        #print('((((((((((((((((((((((', opencood_train_dataset[0]['ego']['sample_idx']) #0
        for i in range(3,len(val_list)):#len(val_list)
            # time1=time.time()
            result_bool = is_in_same_segment(val_list[i], val_list[i-3])
            if result_bool:
                opencood_validate_dataset.change_dataset(i,1)
            else:
                opencood_validate_dataset.change_dataset(i,0)
    elif delay==4:
        opencood_validate_dataset.change_dataset(0,0)
        opencood_validate_dataset.change_dataset(1,0)
        opencood_validate_dataset.change_dataset(2,0)
        opencood_validate_dataset.change_dataset(3,0)

        #print('((((((((((((((((((((((', opencood_train_dataset[0]['ego']['sample_idx']) #0
        for i in range(4,len(val_list)):#len(val_list)
            # time1=time.time()
            result_bool = is_in_same_segment(val_list[i], val_list[i-4])
            if result_bool:
                opencood_validate_dataset.change_dataset(i,1)
            else:
                opencood_validate_dataset.change_dataset(i,0)
    elif delay==10:
        opencood_validate_dataset.change_dataset(0,0)
        opencood_validate_dataset.change_dataset(1,0)
        opencood_validate_dataset.change_dataset(2,0)
        opencood_validate_dataset.change_dataset(3,0)
        opencood_validate_dataset.change_dataset(4,0)
        opencood_validate_dataset.change_dataset(5,0)
        opencood_validate_dataset.change_dataset(6,0)
        opencood_validate_dataset.change_dataset(7,0)
        opencood_validate_dataset.change_dataset(8,0)
        opencood_validate_dataset.change_dataset(9,0)

        #print('((((((((((((((((((((((', opencood_train_dataset[0]['ego']['sample_idx']) #0
        for i in range(10,len(val_list)):#len(val_list)
            # time1=time.time()
            result_bool = is_in_same_segment(val_list[i], val_list[i-10])
            if result_bool:
                opencood_validate_dataset.change_dataset(i,1)
            else:
                opencood_validate_dataset.change_dataset(i,0)

    opencood_dataset=opencood_validate_dataset
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    comm_rates = []

    
    infer_info = opt.fusion_method + opt.note
    ########################################加载model_history start
    opt_history = Namespace(model_dir='opencood/logs/lcfm240602',fusion_method='intermediate')

    assert opt_history.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single', 'late_2model', 'no_inf'] 
    #print(opt1.fusion_method)
    hypes_history = yaml_utils.load_yaml(None, opt_history)
    
    hypes_history['validate_dir'] = hypes_history['test_dir']
    if 'box_align' in hypes_history.keys():
        hypes_history['box_align']['val_result'] = hypes_history['box_align']['test_result']

    #print('Creating Model1') 
    model_history = train_utils.create_model_historybev(hypes_history)
    # we assume gpu is necessary
    #print('Loading Model1 from checkpoint')
    saved_path_history = opt_history.model_dir
    resume_epoch_history , model_history = train_utils.load_saved_model(saved_path_history, model_history)
    if torch.cuda.is_available():
        model_history.cuda()
    model_history.eval()
    ########################################加载model_history end

    for i, batch_data in enumerate(tqdm.tqdm(data_loader)):
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        if batch_data is None:
            continue
        # print('continue',batch_data['ego']['record_len'][0])
        # if batch_data['ego']['record_len'][0]==1:
        #     print('continue')
        #     continue
        with torch.no_grad():

            batch_data = train_utils.to_device(batch_data, device)

            history_data = {
                'processed_lidar': {
                    'voxel_features': batch_data['ego']['history_processed_lidar']['voxel_features'],
                    'voxel_coords': batch_data['ego']['history_processed_lidar']['voxel_coords'],
                    'voxel_num_points': batch_data['ego']['history_processed_lidar']['voxel_num_points']}
                }
            #print('*******************************得到历史bev特征*****************************')
            ouput_dict_history=model_history(history_data)
            history_spatial_features_2d=ouput_dict_history['feature_before_fusion']

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,history_spatial_features_2d,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            elif opt.fusion_method == 'late_2model':
                infer_result = inference_utils.inference_late_fusion_2model(batch_data,
                                                        model, model_inf,
                                                        opencood_dataset)
            elif opt.fusion_method == 'no_inf':
                if len(batch_data) == 1:
                    continue
                infer_result = inference_utils.inference_no_fusion_inf(batch_data,
                                                                model,
                                                                opencood_dataset)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            # fusion_features_2d=infer_result['feature']
            # draw_flag=infer_result['draw_flag']
            if 'comm_rate' in infer_result:
                comm_rates.append(infer_result['comm_rate'])
            # print('画概率密度分布图',fusion_features_2d.shape)

            # print('draw_flag',draw_flag)
            # ####画概率密度分布图
            # if fusion_features_2d.shape[1]==2 and draw_flag==15:
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #     reduced_vehicle_features=fusion_features_2d[:,0].reshape( -1, 256 * 100 * 252).cpu().numpy()
            #     reduced_road_features=fusion_features_2d[:,1].reshape(-1, 256 * 100 * 252).cpu().numpy()
            #     # flattened_features = fusion_features_2d.reshape(2, -1, 256 * 100 * 252)
            #     umap_model = umap.UMAP(n_components=2)
            #     reduced_vehicle_features = umap_model.fit_transform(reduced_vehicle_features)
            #     print(reduced_vehicle_features.shape)

            #     reduced_road_features = umap_model.fit_transform(reduced_road_features)
            #     # 删除包含NaN或inf的行
            #     reduced_vehicle_features = reduced_vehicle_features[~np.isnan(reduced_vehicle_features).any(axis=1)]
            #     reduced_vehicle_features = reduced_vehicle_features[~np.isinf(reduced_vehicle_features).any(axis=1)]

            #     reduced_road_features = reduced_road_features[~np.isnan(reduced_road_features).any(axis=1)]
            #     reduced_road_features = reduced_road_features[~np.isinf(reduced_road_features).any(axis=1)]

            #     kde_vehicle = gaussian_kde(reduced_vehicle_features.T)
            #     kde_road = gaussian_kde(reduced_road_features.T)
            #     x_range = np.linspace(-3, 3, 100)  # 统一设置范围以便于比较
            #     y_range = np.linspace(-3, 3, 100)
            #     X, Y = np.meshgrid(x_range, y_range)
            # # 计算每个点的密度
            #     Z_vehicle = kde_vehicle(np.vstack([X.flatten(), Y.flatten()]))
            #     Z_vehicle = Z_vehicle.reshape(X.shape)

            #     Z_road = kde_road(np.vstack([X.flatten(), Y.flatten()]))
            #     Z_road = Z_road.reshape(X.shape)
                
            #     # # 计算 KL 散度
            #     # p = kde_vehicle(reduced_vehicle_features.T)
            #     # q = kde_road(reduced_road_features.T)
                
            #     # # 避免零概率
            #     # p = np.clip(p, 1e-10, None)
            #     # q = np.clip(q, 1e-10, None)
                
            #     # kl_divergence = np.sum(p * np.log(p / q))

            #     # 绘制概率密度图
            #     plt.figure(figsize=(10, 8))
                
            #     # 绘制车辆特征的概率密度图
            #     plt.contourf(X, Y, Z_vehicle, levels=50, cmap='viridis', alpha=0.5)
            #     plt.colorbar(label='Probability Density')
            #     plt.scatter(reduced_vehicle_features[:, 0], reduced_vehicle_features[:, 1], s=5, alpha=0.5, color='red', label='Vehicle Features')
                
            #     # 绘制路侧特征的概率密度图
            #     plt.contourf(X, Y, Z_road, levels=50, cmap='plasma', alpha=0.5)
            #     plt.scatter(reduced_road_features[:, 0], reduced_road_features[:, 1], s=5, alpha=0.5, color='blue', label='Road Features')

            #     plt.title(f'Probability Density Comparison (Sample {1})')
            #     plt.xlabel('UMAP Dimension 1')
            #     plt.ylabel('UMAP Dimension 2')
            #     plt.legend()
                
            #     plt.tight_layout()
            #     plt.show()
            #     plt.axis("off")

            #     # plt.imshow(canvas.canvas)
            #     plt.tight_layout()
            #     save_path = os.path.join(opt.model_dir, f'KL')

            #     plt.savefig(save_path, transparent=False, dpi=500)
            #     plt.clf()
            #     plt.close()


            # print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq',pred_box_tensor.shape)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "lidar_agent_record": lidar_agent_record})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                """
                If you want 3D visualization, uncomment lines below
                """
                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)
    if len(comm_rates) != 0:
        comm_rates = sum(comm_rates)/len(data_loader)
    print('communication rates: {}'.format(comm_rates))

if __name__ == '__main__':
    main()
