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

			-- All files in source, third_party and applications
        	files {'source/**.hpp',
        		     'applications/' .. name .. '.cpp'
        	}

	        -- Declare the configurations
	        configuration 'Release'
	            targetdir 'build/release'
	            defines {'NDEBUG'}
	            flags {'OptimizeSpeed'}

	        configuration 'Debug'
	            targetdir 'build/debug'
	            defines {'DEBUG'}
	            flags {'Symbols'}

	        configuration 'linux or macosx'
            	includedirs {'/usr/local/include'}
	        	libdirs {'/usr/local/lib'}

	        -- Linux specific settings
	        configuration 'linux'
                links {'pthread'}
	            buildoptions {'-std=c++14'}
	           	linkoptions {'-std=c++14'}

	        -- Mac OS X specific settings
	        configuration 'macosx'
	            buildoptions {'-std=c++14'}
	           	linkoptions {'-std=c++14'}
end
