solution 'vc-GMM'
    configurations {'Release', 'Debug'}
    location 'build'

    for index, file in pairs(os.matchfiles('applications/*.cpp')) do
    	local name = path.getbasename(file)
    	project(name)
    		-- General settings
    		kind 'ConsoleApp'
    		language 'C++'
        	location 'build'

			-- All files in source and applications
        	files {'source/**.hpp',
        		   'applications/' .. name .. '.cpp'
        	}

	        -- Declare the configurations
	        configuration 'Release'
	            targetdir 'build/release'
	            defines {'NDEBUG','STATS_ENABLE_BLAZE_WRAPPERS'}
	            flags {'OptimizeSpeed'}

	        configuration 'Debug'
	            targetdir 'build/debug'
	            defines {'DEBUG','STATS_ENABLE_BLAZE_WRAPPERS'}
	            flags {'Symbols'}

	        configuration 'linux or macosx'
            	includedirs {'/usr/local/include'}
	        	libdirs {'/usr/local/lib'}
                links('cnpy')

	        -- Linux specific settings
	        configuration 'linux'
                links {'pthread'}
	            buildoptions {'-std=c++17'}
	           	linkoptions {'-std=c++17'}

	        -- Mac OS X specific settings
	        configuration 'macosx'
	            buildoptions {'-std=c++17'}
	           	linkoptions {'-std=c++17'}
end
